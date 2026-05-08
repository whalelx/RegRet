import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# project imports
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)

from collators.qwen2_vision_process import process_vision_info_with_focal
from loaders.processor import LemuirProcessor
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration


BOX_OP = os.environ.get("BOX_OP", "crop")


def add_embed_token(tokenizer, model, emb_token: str = "<emb>"):
    num_new_tokens = tokenizer.add_tokens([emb_token])
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.config.emb_token_ids = tokenizer.convert_tokens_to_ids([emb_token])


def compute_map_at_k(similarity, true_indices: List[List[int]], k: int = 50):
    """Compute mAP@K for a similarity matrix and ground-truth indices."""
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.tensor(similarity)
    device = similarity.device
    num_queries = similarity.shape[0]

    _, topk_indices = torch.topk(similarity, k=min(k, similarity.shape[1]), dim=1)

    average_precisions = []
    for i in range(num_queries):
        pred_indices = topk_indices[i]
        gt_indices = true_indices[i]

        if not isinstance(gt_indices, torch.Tensor):
            gt_indices = torch.tensor(gt_indices, device=device, dtype=torch.long)
        else:
            gt_indices = gt_indices.to(device)

        num_gt = len(gt_indices)
        if num_gt == 0:
            average_precisions.append(0.0)
            continue

        hits = torch.isin(pred_indices, gt_indices) if hasattr(torch, "isin") else (
            (pred_indices.unsqueeze(1) == gt_indices.unsqueeze(0)).any(dim=1)
        )
        if hits.sum() == 0:
            average_precisions.append(0.0)
            continue

        hits = hits.float()
        cumsum_hits = torch.cumsum(hits, dim=0)
        ranks = torch.arange(1, hits.shape[0] + 1, device=device, dtype=torch.float)
        precision_at_i = cumsum_hits / ranks

        score = (precision_at_i * hits).sum() / num_gt
        average_precisions.append(score.item())

    return float(sum(average_precisions) / len(average_precisions)) if average_precisions else 0.0


def compute_recall_at_1(similarity, true_indices: List[List[int]]):
    """Recall@1: fraction of queries whose top-1 is a ground-truth match."""
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.tensor(similarity)
    device = similarity.device
    num_queries = similarity.shape[0]

    top1 = torch.argmax(similarity, dim=1)
    hits = []
    for i in range(num_queries):
        gt = true_indices[i]
        if not isinstance(gt, torch.Tensor):
            gt = torch.tensor(gt, device=device, dtype=torch.long)
        else:
            gt = gt.to(device)
        if gt.numel() == 0:
            hits.append(0.0)
            continue
        hits.append(float((top1[i] == gt).any().item()))
    return float(np.mean(hits)) if hits else 0.0


def _normalize_box(bbox: List[float], w: float, h: float) -> List[float]:
    if not bbox or w <= 0 or h <= 0:
        return [0.0, 0.0, 1.0, 1.0]
    x, y, bw, bh = bbox
    return [x / w, y / h, (x + bw) / w, (y + bh) / h]


def load_reircoco(json_path: str, images_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse ReIRCOCO-style JSON (not jsonl).

    Expected keys:
    - images: list of {id, file_name, width, height, expressions(optional list[str])}
    - annotations: list of {image_id, bbox, ...}

    Returns:
        db_items: list of {image_path, box, key}
        query_items: list of {text, key}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    # collect first bbox per image
    bbox_map: Dict[int, List[float]] = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is None:
            continue
        if "bbox" in ann and ann["bbox"]:
            bbox_map.setdefault(img_id, []).append(ann["bbox"])

    db_items: List[Dict] = []
    query_items: List[Dict] = []

    for img in images:
        img_id = img.get("id")
        file_name = img.get("file_name")
        expressions = img.get("expressions")[0:1] # HACK follow objemb, 只取了第一个exp进行评估

        img_path = os.path.join(images_dir, file_name)
        bbox = None
        if img_id in bbox_map:
            bbox = bbox_map[img_id][0]
        norm_box = _normalize_box(bbox, img.get("width"), img.get("height"))

        key = str(img_id) if img_id is not None else file_name

        # breakpoint()
        db_items.append({"image_path": img_path, "box": norm_box, "key": key})

        for exp in expressions:
            if exp:
                query_items.append({"text": exp, "key": key})

    return db_items, query_items


class ReIRCOCODataset(Dataset):
    def __init__(self, items: List[Dict], mode: str):
        self.items = items
        self.mode = mode  # "db" or "query"
        self.keys = [it["key"] for it in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.mode == "db":
            image = Image.open(item["image_path"]).convert("RGB")
            return {
                "image": image,
                "box": item.get("box", [0.0, 0.0, 1.0, 1.0]),
                "text": None,
                "key": item["key"],
            }
        else:  # query (text only)
            return {
                "image": None,
                "box": None,
                "text": item["text"],
                "key": item["key"],
            }


def collate_fn(batch, processor, tokenizer, mode: str = "query_text"):
    """
    mode: "query_text" (text query) or "db" (image database)
    """
    messages_list = []

    is_text_mode = mode == "query_text"
    for item in batch:
        if is_text_mode:
            raw_text = item.get("text", "") or ""
            text_prompt = (
                # f"Find me an everyday image that matches the given caption. {raw_text}\nSummarize above sentence in one word: "
                f"Find the object described by the text: {raw_text}\nSummarize above sentence in one word: "
                # f"Find me an everyday image with a region(which is appended as the second image) that matches the given caption. {raw_text}\nSummarize above sentence in one word: "
            
            )
            msgs = [
                {"role": "user", "content": [{"type": "text", "text": text_prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]
        else:
            text_prompt = "\nSummarize above image in one word: "
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": item["image"],
                            "box": item.get("box"),
                            "box_op": BOX_OP,
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
            ]
        messages_list.append(msgs)

    image_inputs, id_dict = process_vision_info_with_focal(messages_list, box_op=BOX_OP)
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]

    replace_two_imgs = set(id_dict.multi_img_texts) if hasattr(id_dict, "multi_img_texts") else set()
    inputs, crop_or_concat_img_inputs = processor(
        text=texts,
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
        id_dict=id_dict,
        replace_two_imgs=replace_two_imgs,
    )

    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": None,
        "labels": labels,
    }

    if not is_text_mode:
        model_inputs["image_grid_thw"] = inputs.get("image_grid_thw")
        model_inputs.update(crop_or_concat_img_inputs)

    keys = [item["key"] for item in batch]
    return model_inputs, id_dict, keys


def get_embeddings(dataloader, model, accelerator):
    model.eval()
    all_embeds = []
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        model_inputs, id_dict, _ = batch
        model_inputs = {
            k: v.to(accelerator.device)
            for k, v in model_inputs.items()
            if v is not None and isinstance(v, torch.Tensor)
        }

        with torch.no_grad():
            embeds = model(**model_inputs, id_dict=id_dict, inference=True)
            embeds = F.normalize(embeds, dim=-1)
            gathered_embeds = accelerator.gather_for_metrics(embeds)
            all_embeds.append(gathered_embeds.cpu())

    return torch.cat(all_embeds, dim=0) if all_embeds else torch.empty(0)


def build_true_indices(query_keys: List[str], db_keys: List[str]) -> List[List[int]]:
    mapping = defaultdict(list)
    for idx, k in enumerate(db_keys):
        mapping[k].append(idx)
    return [mapping.get(qk, []) for qk in query_keys]


def run_eval(args):
    accelerator = Accelerator()

    accelerator.print("Loading annotations...")
    db_items, query_items = load_reircoco(args.jsonl_path, args.images_dir)
    accelerator.print(f"DB images: {len(db_items)}, Queries: {len(query_items)}")

    db_dataset = ReIRCOCODataset(db_items, mode="db")
    query_dataset = ReIRCOCODataset(query_items, mode="query")

    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    processor = LemuirProcessor.from_pretrained(args.original_model_id or args.model_id)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"

    add_embed_token(processor.tokenizer, model)

    db_loader = DataLoader(
        db_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor, processor.tokenizer, "db"),
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor, processor.tokenizer, "query_text"),
    )

    model, db_loader, query_loader = accelerator.prepare(model, db_loader, query_loader)

    query_embeds = get_embeddings(query_loader, model, accelerator)
    db_embeds = get_embeddings(db_loader, model, accelerator)

    if accelerator.is_main_process:
        db_keys = db_dataset.keys
        query_keys = query_dataset.keys

        if db_embeds.numel() == 0 or query_embeds.numel() == 0:
            accelerator.print("No embeddings produced. Check data paths and annotations.")
            return

        db_embeds = db_embeds.to(accelerator.device)
        query_embeds = query_embeds.to(accelerator.device)
        scores = torch.matmul(query_embeds, db_embeds.t())

        true_indices = build_true_indices(query_keys, db_keys)
        # map_score = compute_map_at_k(scores, true_indices, k=args.map_k)
        recall1 = compute_recall_at_1(scores, true_indices)

        accelerator.print("\nFinal Results (ReIRCOCO text->image):")
        accelerator.print(f"Total Queries: {len(query_keys)}")
        accelerator.print(f"Recall@1: {recall1:.4f}")
        # accelerator.print(f"mAP@{args.map_k}: {map_score:.4f}")

        # show top-10 candidates for the first query
        if len(query_keys) > 0:
            first_query_idx = 0
            first_query_text = query_items[first_query_idx]["text"]
            topk_scores, topk_indices = torch.topk(
                scores[first_query_idx], k=min(10, scores.shape[1])
            )

            key_to_path = {item["key"]: item["image_path"] for item in db_items}
            accelerator.print("\nTop-10 candidates for the first query:")
            accelerator.print(f"Query: {first_query_text}")
            for rank, (idx, sc) in enumerate(zip(topk_indices.tolist(), topk_scores.tolist()), start=1):
                cand_key = db_keys[idx]
                cand_path = key_to_path.get(cand_key, "N/A")
                accelerator.print(
                    f"{rank:02d}. id={cand_key}, score={sc:.4f}, image_path={cand_path}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to ReIRCOCO JSON annotations (with images/annotations)")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing COCO images")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--original_model_id", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--map_k", type=int, default=50)
    args = parser.parse_args()
    args.jsonl_path = getattr(args, "json_path", None)  # backward compat
    run_eval(args)
