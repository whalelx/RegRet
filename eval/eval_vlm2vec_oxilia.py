import json
import sys
import os
import pickle
import torch
import time
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from transformers import AutoConfig
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from dataset.datasets_mbeir import QueryDataset, CandidateDataset

from vlm2vec.src.arguments import ModelArguments, DataArguments, TrainingArguments
from vlm2vec.src.model import MMEBModel
from vlm2vec.src.collator import EvalCollator
from vlm2vec.src.model_utils import get_backbone_name

TOKEN = "<image>"

DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000

def unhash_qid(hashed_qid):
    dataset_id = hashed_qid // DATASET_QUERY_NUM_UPPER_BOUND
    data_within_id = hashed_qid % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def unhash_did(hashed_did):
    dataset_id = hashed_did // DATASET_CAN_NUM_UPPER_BOUND
    data_within_id = hashed_did % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{data_within_id}"

def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if int(relevance_score) > 0:  # Assuming only positive relevance scores indicate relevant documents
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(
        f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid

def compute_recall_at_k_rewrite(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0 # Return 0 if there are no relevant documents

    relevant_docs_set = set(relevant_docs)
    retrieved_indices_set = set(retrieved_indices)

    result = []
    for target_doc in relevant_docs_set:
        other_relevant_docs = relevant_docs_set - {target_doc}
        num_other_relevant_retrieved = len(retrieved_indices_set.intersection(other_relevant_docs))
        dynamic_k = k + num_other_relevant_retrieved
        top_dynamic_k_retrieved_set = set(retrieved_indices[:dynamic_k])
        
        if target_doc in top_dynamic_k_retrieved_set:
            result.append(1.0)
        else:
            result.append(0.0)
            
    return result

from eval_clip import SiglipCandidateDataCollator
class MME5CandidateDataCollator(SiglipCandidateDataCollator):
    def __init__(self, image_path_prefix, cand_modal="image,text", image_size=None, processor=None):
        self.processor = processor
        self.image_path_prefix = image_path_prefix
        self.cand_modal = cand_modal.split(',') if cand_modal else ["image", "text"]
        self.use_image = "image" in self.cand_modal
        self.use_text = "text" in self.cand_modal
        self.image_size = image_size

    def __call__(self, batch):
        images = []
        texts = []
        candidate_ids = []
        boxinfo_tensors = []
        
        for item in batch:
            # item is a tuple: (candidate_message, did)
            candidate_message, did = item
            candidate_ids.append(did)
            
            # 提取文本、图片、box和box_op信息
            text_content = ""
            image_path = None
            box = None
            box_op = None
            
            # 从消息中提取内容
            if candidate_message and len(candidate_message) > 0:
                user_content = candidate_message[0].get('content', [])
                for content_item in user_content:
                    if content_item.get('type') == 'text':
                        text_content = content_item.get('text', '')
                    elif content_item.get('type') == 'image':
                        image_path = content_item.get('image', None)
                        box = content_item.get('box', None)
                        box_op = content_item.get('box_op', None)

            text_content = text_content.replace("\nSummarize above sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image and sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image in one word: ", "")
            
            if self.use_image and image_path:
                image = Image.open(image_path).convert('RGB')
                
                image = self.process_image_with_box_op(image, box, box_op)
                images.append(image)
                text_content = f"{TOKEN}\nRepresent the given image with related text information:" + text_content
            else:
                images.append(None)

            # 根据modal设置决定是否使用文本
            if self.use_text:
                texts.append(text_content)
            else:
                texts.append(f"{TOKEN}\nRepresent the given image.")  # 空文本
        
        return [ (i,j,k) for i,j,k in zip(texts, images, candidate_ids) ]

from eval_clip import SiglipCandidateDataCollator
class MME5QueryDataCollator(SiglipCandidateDataCollator):
    def __init__(self, image_path_prefix, cand_modal="image,text", image_size=None, processor=None):
        self.processor = processor
        self.image_path_prefix = image_path_prefix
        self.cand_modal = cand_modal.split(',') if cand_modal else ["image", "text"]
        self.use_image = "image" in self.cand_modal
        self.use_text = "text" in self.cand_modal
        self.image_size = image_size

    def __call__(self, batch):
        images = []
        texts = []
        candidate_ids = []
        boxinfo_tensors = []
        
        for item in batch:
            # item is a tuple: (candidate_message, did)
            candidate_message, did = item
            candidate_ids.append(did)
            
            # 提取文本、图片、box和box_op信息
            text_content = ""
            image_path = None
            box = None
            box_op = None
            
            # 从消息中提取内容
            if candidate_message and len(candidate_message) > 0:
                user_content = candidate_message[0].get('content', [])
                for content_item in user_content:
                    if content_item.get('type') == 'text':
                        text_content = content_item.get('text', '')
                    elif content_item.get('type') == 'image':
                        image_path = content_item.get('image', None)
                        box = content_item.get('box', None)
                        box_op = content_item.get('box_op', None)

            text_content = text_content.replace("\nSummarize above sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image and sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image in one word: ", "")

            if self.use_image and image_path:
                image = Image.open(image_path).convert('RGB')
                
                image = self.process_image_with_box_op(image, box, box_op)
                images.append(image)
                text_content = f"{TOKEN}\n" + text_content  # 改用 llava_next 的图像 token
            else:
                images.append(None)

            # 根据modal设置决定是否使用文本
            if self.use_text:
                texts.append(text_content)
            else:
                texts.append(f"{TOKEN}\nRepresent the given image.")  # 空文本

        return [ (i,j,k) for i,j,k in zip(texts, images, candidate_ids) ]


def _setup_model_and_collator(args):
    """
    Shared model / processor / eval_collator setup for all eval tasks.
    Returns (model, eval_collator, model_args, training_args, processor).
    """
    model_args = ModelArguments(model_name=args.model_name)
    data_args  = DataArguments()
    training_args = TrainingArguments()

    model_args.model_name = args.model_name
    model_args.normalize  = args.normalize
    model_args.pooling    = args.pooling
    model_args.num_crops  = getattr(args, 'num_crops', 4)

    data_args.image_dir = getattr(args, 'image_path_prefix', '')
    data_args.max_len   = 1024

    training_args.per_device_eval_batch_size = getattr(args, 'batch_size', 32)

    # Auto-detect backbone via AutoConfig
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    model_args.model_backbone = model_backbone
    training_args.dataloader_num_workers = 0

    from transformers import LlavaNextProcessor
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        trust_remote_code=True,
    )
    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)
    eval_collator = EvalCollator(
        data_args=data_args, model_args=model_args, processor=processor,
    )

    return model, eval_collator, model_args, training_args, processor


def _encode_dataloader(dataloader, model, device, role="qry"):
    """
    Encode a dataloader and return (features, ids).
    role: "qry" or "tgt"  (determines which model head is used).
    """
    all_features = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding ({role})"):
            batch_ids = batch.get('qiddid', [])
            if isinstance(batch_ids, torch.Tensor):
                all_ids.extend(batch_ids.cpu().numpy().tolist())
            else:
                all_ids.extend(batch_ids)
            batch = {k: v.to(device) for k, v in batch.items() if k != 'qiddid'}

            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                if role == "qry":
                    emb = model(qry=batch)["qry_reps"]
                else:
                    emb = model(tgt=batch)["tgt_reps"]
                all_features.append(emb)

    return torch.cat(all_features, dim=0), all_ids


# ==================== iLIAS Evaluation ====================

class iIiasDatasetForMME5(Dataset):
    """Wraps the HuggingFace iLIAS dataset for vlm2vec evaluation."""
    def __init__(self, hf_split, is_query=True, mode="image"):
        self.data = hf_split
        self.is_query = is_query
        self.mode = mode  # "image" or "text"
        
        # Build string key to numeric ID mapping (without loading images)
        self.key_to_id = {}
        self.id_to_key = {}
        # Use index access to avoid triggering image decoding
        for idx in range(len(self.data)):
            # Access raw key without triggering full decode
            key = self.data[idx]['__key__']
            self.key_to_id[key] = idx
            self.id_to_key[idx] = key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use numeric ID directly
        numeric_id = idx
        
        item = self.data[idx]

        if self.mode == "text":
            txt_content = item.get('txt', '')
            return (txt_content, None, numeric_id)

        # image mode
        image = item['jpg'].convert("RGB")
        bbox_list = item.get('bbox.json', [])
        if bbox_list and len(bbox_list) > 0:
            x, y, w_box, h_box = bbox_list[0]
            img_w, img_h = image.size
            box = [x / img_w, y / img_h, (x + w_box) / img_w, (y + h_box) / img_h]
        else:
            box = [0.0, 0.0, 1.0, 1.0]

        return (image, box, numeric_id)


class IliasCollatorForMME5:
    """Collator that converts iLIAS items to (text, image, id) tuples for vlm2vec."""
    def __init__(self, mode="query", box_op="crop"):
        self.mode = mode
        self.box_op = box_op

    def _crop_image(self, image, box):
        if box is None:
            return image
        w, h = image.size
        x0, y0, x1, y1 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            return image.crop((x0, y0, x1, y1))
        return image

    def __call__(self, batch):
        results = []
        for item in batch:
            if self.mode == "query_text":
                # item = (text, None, numeric_id)
                raw_text, _, numeric_id = item
                # Text-only query: no image token needed
                text_content = f"Find me an image that matches the given caption. {raw_text}" if raw_text else ""
                results.append((text_content, None, numeric_id))
            elif self.mode == "query":
                # item = (image, box, numeric_id)
                image, box, numeric_id = item
                image = self._crop_image(image, box)
                text_content = f"{TOKEN}\nFind the most similar image."
                results.append((text_content, image, numeric_id))
            else:
                # db mode: item = (image, box, numeric_id)
                image, box, numeric_id = item
                # db images: no crop
                text_content = f"{TOKEN}\nRepresent the given image."
                results.append((text_content, image, numeric_id))
        return results


def compute_map_at_k(similarity, true_indices, k=50):
    """Compute mAP@K (from ilias.py)."""
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.tensor(similarity)
    device = similarity.device
    num_queries = similarity.shape[0]
    _, topk_indices = torch.topk(similarity, k=k, dim=1)
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
        if hasattr(torch, 'isin'):
            hits = torch.isin(pred_indices, gt_indices)
        else:
            hits = (pred_indices.unsqueeze(1) == gt_indices.unsqueeze(0)).any(dim=1)
        if hits.sum() == 0:
            average_precisions.append(0.0)
            continue
        hits = hits.float()
        cumsum_hits = torch.cumsum(hits, dim=0)
        ranks = torch.arange(1, k + 1, device=device, dtype=torch.float)
        precision_at_i = cumsum_hits / ranks
        score = (precision_at_i * hits).sum() / num_gt
        average_precisions.append(score.item())
    if not average_precisions:
        return 0.0
    return sum(average_precisions) / len(average_precisions)


def eval_ilias(args):
    """Evaluate iLIAS dataset with vlm2vec model."""
    from datasets import load_dataset as hf_load_dataset

    print(f"Starting iLIAS evaluation with model: {args.model_name}")
    print(f"Eval mode: {args.ilias_eval_mode}")

    # ----- Setup model -----
    model, eval_collator, model_args, training_args, processor = _setup_model_and_collator(args)

    # ----- Load iLIAS datasets -----
    print("Loading database dataset...")
    db_hf = hf_load_dataset(args.ilias_dataset_path, name="core_db", split="core_db")
    db_dataset = iIiasDatasetForMME5(db_hf, is_query=False, mode="image")
    # Reuse the key_to_id mapping from db_dataset
    db_key_to_id = db_dataset.key_to_id
    
    # Pre-build prefix to db ids mapping for efficient GT lookup
    print("Building database prefix index...")
    db_prefix_to_ids = {}
    for db_key, db_idx in db_key_to_id.items():
        # db_key format: "bold_bimp_000/pos/P000_07" or similar
        # Extract prefix (e.g., "bold_bimp_000")
        if '/' in db_key:
            prefix = db_key.split('/')[0]
        else:
            prefix = db_key
        if prefix not in db_prefix_to_ids:
            db_prefix_to_ids[prefix] = []
        db_prefix_to_ids[prefix].append(db_idx)
    
    print(f"  Loaded {len(db_dataset)} database items with {len(db_prefix_to_ids)} unique prefixes")

    print("Loading query dataset...")
    if args.ilias_eval_mode == "text":
        query_hf = hf_load_dataset(args.ilias_dataset_path, name="text_queries", split="text_queries")
        query_dataset = iIiasDatasetForMME5(query_hf, is_query=True, mode="text")
        query_collate_mode = "query_text"
    else:
        query_hf = hf_load_dataset(args.ilias_dataset_path, name="img_queries", split="img_queries")
        query_dataset = iIiasDatasetForMME5(query_hf, is_query=True, mode="image")
        query_collate_mode = "query"
    
    query_key_to_id = query_dataset.key_to_id
    print(f"  Loaded {len(query_dataset)} queries")

    # Build ground truth mapping: query string key -> list of positive db numeric ids
    print("Building ground truth mappings...")
    query_key_to_pos_db_ids = {}
    for q_idx in range(len(query_hf)):
        # Access key without triggering full decode
        q_key = query_hf[q_idx]['__key__']
        try:
            prefix = q_key.split('/query/')[0]  # e.g., "bold_bimp_000"
        except IndexError:
            continue
        
        # Find matching db items by prefix
        pos_db_ids = db_prefix_to_ids.get(prefix, [])
        query_key_to_pos_db_ids[q_key] = pos_db_ids
    
    print(f"  Built GT for {len(query_key_to_pos_db_ids)} queries")

    query_data_collator = IliasCollatorForMME5(mode=query_collate_mode)
    db_data_collator = IliasCollatorForMME5(mode="db")

    batch_size = training_args.per_device_eval_batch_size
    num_workers = training_args.dataloader_num_workers
    query_dataloader = DataLoader(
        query_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=lambda batch: eval_collator(query_data_collator(batch))
    )
    db_dataloader = DataLoader(
        db_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=lambda batch: eval_collator(db_data_collator(batch))
    )

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device

    # ----- Encode -----
    query_features, query_ids = _encode_dataloader(query_dataloader, model, device, role="qry")
    db_features, db_ids       = _encode_dataloader(db_dataloader, model, device, role="tgt")

    if accelerator.is_main_process:
        # Build ground truth using numeric IDs - aligned with query_ids order
        # query_ids are numeric IDs, need to map back to string keys for GT lookup
        print("Finalizing ground truth for evaluation...")
        true_indices = []
        for q_numeric_id in tqdm(query_ids, desc="Building GT"):
            # Map numeric ID back to string key
            q_key = query_dataset.id_to_key[q_numeric_id]
            # Get the positive db numeric IDs for this query
            gt_idx = query_key_to_pos_db_ids.get(q_key, [])
            true_indices.append(gt_idx)

        scores = torch.matmul(query_features.to(device), db_features.to(device).t())

        map_score = compute_map_at_k(scores, true_indices, k=50)
        print(f"\n===== iLIAS Results (Mode: {args.ilias_eval_mode}) =====")
        print(f"Total Queries: {len(query_ids)}")
        print(f"mAP@50: {map_score:.4f}")

        save_dir = "./SigLIP_Ret_eval_results"
        os.makedirs(save_dir, exist_ok=True)
        model_short = args.model_name.split('/')[-1]
        with open(f"{save_dir}/{model_short}_ilias_results.txt", 'a') as f:
            f.write(f"iLIAS eval_mode={args.ilias_eval_mode}\n")
            f.write(f"mAP@50 = {map_score:.4f}\n")


# ==================== Oxford / Paris Evaluation ====================

class OxfordDatasetForMME5(Dataset):
    """Wraps the HuggingFace Oxford/Paris dataset for vlm2vec evaluation."""
    def __init__(self, hf_split, use_bbox=False):
        self.data = hf_split
        self.use_bbox = use_bbox

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"].convert("RGB")
        img_w, img_h = image.size
        if self.use_bbox and item["bbx"]:
            x1, y1, x2, y2 = item["bbx"]
            box = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]
        else:
            box = [0.0, 0.0, 1.0, 1.0]
        return (image, box, idx)


class OxfordCollatorForMME5:
    """Collator that converts Oxford/Paris items to (text, image, id) tuples."""
    def __init__(self, mode="db", box_op="crop"):
        self.mode = mode
        self.box_op = box_op

    def _crop_image(self, image, box):
        if box is None:
            return image
        w, h = image.size
        x0, y0, x1, y1 = box[0]*w, box[1]*h, box[2]*w, box[3]*h
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            return image.crop((x0, y0, x1, y1))
        return image

    def __call__(self, batch):
        results = []
        for item in batch:
            image, box, item_id = item
            if self.mode == "query":
                image = self._crop_image(image, box)
                text_content = f"{TOKEN}\nFind the most similar image."
            else:
                text_content = f"{TOKEN}\nRepresent the given image."
            results.append((text_content, image, item_id))
        return results


def compute_oxford_map(similarity, query_dataset, num_db):
    """Oxford/Paris revisit protocol mAP (from eval_oxford.py)."""
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.tensor(similarity, dtype=torch.float32)
    ranks = torch.argsort(similarity, dim=1, descending=True).T.numpy()
    nq = len(query_dataset)
    aps = []
    nempty = 0
    for i in range(nq):
        query_info = query_dataset[i]
        qgnd = np.array(list(query_info["easy"]) + list(query_info["hard"]), dtype=np.int64)
        qgndj = np.array(list(query_info["junk"]), dtype=np.int64)
        if qgnd.shape[0] == 0:
            nempty += 1
            continue
        pos = np.where(np.isin(ranks[:, i], qgnd))[0]
        junk = np.where(np.isin(ranks[:, i], qgndj))[0]
        k = 0
        ij = 0
        if len(junk):
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] -= k
                ip += 1
        ap = 0.0
        for j, p in enumerate(pos):
            ap += (j + 1) / (p + 1)
        ap /= len(qgnd)
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def eval_oxford(args):
    """Evaluate Oxford5k / Paris6k with vlm2vec model."""
    from datasets import load_dataset as hf_load_dataset

    dataset_name = getattr(args, 'oxford_dataset_name', 'roxford5k')
    print(f"Starting Oxford/Paris evaluation with model: {args.model_name}")
    print(f"Dataset: {dataset_name}")

    # ----- Setup model -----
    model, eval_collator, model_args, training_args, processor = _setup_model_and_collator(args)

    # ----- Load Oxford/Paris datasets -----
    dataset_script = args.oxford_dataset_script  # e.g. "dataset/dataset_oxford5k.py"
    query_hf = hf_load_dataset(dataset_script, name=dataset_name, split="qimlist", trust_remote_code=True)
    db_hf = hf_load_dataset(dataset_script, name=dataset_name, split="imlist", trust_remote_code=True)

    query_dataset = OxfordDatasetForMME5(query_hf, use_bbox=True)
    db_dataset = OxfordDatasetForMME5(db_hf, use_bbox=False)

    box_op = getattr(args, 'oxford_box_op', 'crop')
    query_data_collator = OxfordCollatorForMME5(mode="query", box_op=box_op)
    db_data_collator = OxfordCollatorForMME5(mode="db", box_op=box_op)

    batch_size = training_args.per_device_eval_batch_size
    num_workers = training_args.dataloader_num_workers
    query_dataloader = DataLoader(
        query_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=lambda batch: eval_collator(query_data_collator(batch))
    )
    db_dataloader = DataLoader(
        db_dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, collate_fn=lambda batch: eval_collator(db_data_collator(batch))
    )

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device

    # ----- Encode -----
    query_features, _ = _encode_dataloader(query_dataloader, model, device, role="qry")
    db_features, _    = _encode_dataloader(db_dataloader, model, device, role="tgt")

    if accelerator.is_main_process:
        scores = torch.matmul(query_features.to(device), db_features.to(device).t())
        mAP = compute_oxford_map(scores.cpu(), query_hf, num_db=len(db_hf))

        print(f"\n===== Oxford/Paris Results on {dataset_name} =====")
        print(f"Queries : {len(query_hf)}")
        print(f"Gallery : {len(db_hf)}")
        print(f"mAP     : {mAP:.4f}")

        save_dir = "./SigLIP_Ret_eval_results"
        os.makedirs(save_dir, exist_ok=True)
        model_short = args.model_name.split('/')[-1]
        with open(f"{save_dir}/{model_short}_oxford_results.txt", 'a') as f:
            f.write(f"dataset={dataset_name}\n")
            f.write(f"mAP = {mAP:.4f}\n")


# ==================== Original MMEB Evaluation ====================

def eval_mmeb(args):
    print(f"Starting MMEB evaluation with model: {args.model_name}")
    print(f"Batch size: {getattr(args, 'batch_size', 32)}")

    # ----- Setup model -----
    model, eval_collator, model_args, training_args, processor = _setup_model_and_collator(args)

    # Load datasets
    query_dataset = QueryDataset(
        query_data_path=args.query_data_path, 
        cand_pool_path=args.query_cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix
    )

    cand_dataset = CandidateDataset(
        query_data_path=args.query_data_path, 
        cand_pool_path=args.cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix
    )

    query_data_collator = MME5QueryDataCollator(image_path_prefix=args.image_path_prefix, cand_modal=args.query_modal, processor=processor)
    cand_data_collator = MME5CandidateDataCollator(image_path_prefix=args.image_path_prefix, cand_modal=args.cand_modal, processor=processor)
    # Create dataloaders
    batch_size = training_args.per_device_eval_batch_size
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        num_workers=training_args.dataloader_num_workers,
        shuffle=False,
        collate_fn=lambda batch: eval_collator(query_data_collator(batch))
    )
    
    candidate_dataloader = DataLoader(
        cand_dataset,
        batch_size=batch_size,
        num_workers=training_args.dataloader_num_workers,
        shuffle=False,
        collate_fn=lambda batch: eval_collator(cand_data_collator(batch))
    )
    
    # Setup accelerator
    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    is_main_process = accelerator.is_main_process
    
    # Encode queries and candidates
    query_features, query_ids       = _encode_dataloader(query_dataloader, model, device, role="qry")
    candidate_features, candidate_ids = _encode_dataloader(candidate_dataloader, model, device, role="tgt")
    
    if is_main_process:
        import numpy as np 

        index = []
        scores = []
        for i in range(len(query_features)):
            query_feature = query_features[i:i+1]
            score = query_feature @ candidate_features.T # (1, num_candidate)
            topk_score, topk_indexes = torch.topk(score, k=50, dim=-1)
            topk_indexes = topk_indexes.squeeze().tolist()
            index.append(topk_indexes)
            scores.append(topk_score.tolist())

        cand_names = np.array([[unhash_did(candidate_ids[item]) for item in row] for row in index])
        query_names = [unhash_qid(item) for item in query_ids]


        save_dir_name = "./SigLIP_Ret_eval_results"
        if not os.path.exists(save_dir_name):
            os.makedirs(save_dir_name)
        save_name = args.qrels_path.split('/')[-1].replace('_qrels.txt', '')
        model_name = args.model_name.split('/')[-1]
        save_name = f"{save_name}_{model_name}"
        # with open(f"{save_dir_name}/{save_name}_query_names.json", 'w') as f:
        #     json.dump(query_names, f, indent=2)
        # with open(f"{save_dir_name}/{save_name}_cand_names.json", 'w') as f:
        #     json.dump(cand_names.tolist(), f, indent=2)
        # with open(f"{save_dir_name}/{save_name}_scores.json", 'w') as f:
        #     json.dump(scores, f, indent=2)
        

        qrel, qid_to_taskid = load_qrel(args.qrels_path)
        
        k_lists = [1, 5, 10,50]
        res = {}
        
        for k in k_lists:
            res[f'recall_{k}'] = []

        for ind, query_name in enumerate(tqdm(query_names)):
            relevant_docs = qrel[query_name]
            retrieved_indices_for_qid = cand_names[ind]
            for k in k_lists:
                recall_at_k = compute_recall_at_k_rewrite(relevant_docs, retrieved_indices_for_qid, k)
                res[f'recall_{k}'].extend(recall_at_k)

        for k in k_lists:
            print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}")

        model_name = model_args.model_name.split('/')[-1]
        with open(f"{save_dir_name}/{model_name}_results.txt", 'a') as f:
            f.write(args.qrels_path + '\n')
            for k in k_lists:
                f.write(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}" + '\n')
        print(f"Write output to {save_dir_name}/{model_name}_results.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate retrieval performance using vlm2vec model')
    parser.add_argument('--eval_task', type=str, default='mmeb', choices=['mmeb', 'ilias', 'oxford'],
                        help='Evaluation task: "mmeb", "ilias", or "oxford"')

    # Common args
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num_crops', type=int, default=4, help='Number of crops for processor')
    parser.add_argument("--pooling", type=str, default="last", help="Pooling method")
    parser.add_argument("--normalize", type=bool, default=True, help="Whether to normalize features")
    parser.add_argument('--image_path_prefix', type=str, default='', help='Prefix path for images')

    # MMEB-specific args
    parser.add_argument('--query_data_path', type=str, default=None, help='Path to query data')
    parser.add_argument('--cand_pool_path', type=str, default=None, help='Path to candidate pool data')
    parser.add_argument('--instructions_path', type=str, default=None, help='Path to instructions')
    parser.add_argument('--qrels_path', type=str, default=None, help='Path to qrels file')
    parser.add_argument('--query_cand_pool_path', type=str, default=None, help='Path to query candidate pool')
    parser.add_argument('--query_modal', type=str, default='image,text',
                        help='Modalities for query: "image", "text", or "image,text"')
    parser.add_argument('--cand_modal', type=str, default='image,text',
                        help='Modalities for candidate: "image", "text", or "image,text"')

    # iLIAS-specific args
    parser.add_argument('--ilias_dataset_path', type=str, default=None,
                        help='HF dataset path for iLIAS (local dir or hub)')
    parser.add_argument('--ilias_eval_mode', type=str, default='image', choices=['image', 'text'],
                        help='iLIAS eval mode: "image" or "text"')

    # Oxford/Paris-specific args
    parser.add_argument('--oxford_dataset_script', type=str, default='dataset/dataset_oxford5k.py',
                        help='Path to Oxford/Paris HF dataset script')
    parser.add_argument('--oxford_dataset_name', type=str, default='roxford5k',
                        choices=['roxford5k', 'rparis6k'], help='Oxford/Paris dataset name')
    parser.add_argument('--oxford_box_op', type=str, default='crop',
                        help='Box operation for Oxford queries: crop/draw/none')

    args = parser.parse_args()

    if args.eval_task == 'ilias':
        eval_ilias(args)
    elif args.eval_task == 'oxford':
        eval_oxford(args)
    else:
        eval_mmeb(args)

