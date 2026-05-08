"""
Evaluate RzenEmbed model on the same datasets as eval_vlm2vec_oxfilia.py:
  - MMEB (M-BEIR) retrieval
  - iLIAS image/text retrieval
  - Oxford5k / Paris6k image retrieval

Supports multi-GPU parallel encoding via torch.distributed.
Launch with:
  torchrun --nproc_per_node=NUM_GPUS eval/eval_rzen.py [args...]
"""
import json
import os
import sys
import math
import torch
import torch.distributed as dist
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)


from eval.model_plugins.rzen import RzenEmbed

current_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_path)

from dataset.datasets_mbeir import QueryDataset, CandidateDataset
from eval_clip import SiglipCandidateDataCollator


BOX_OP = "draw"

# ======================== Distributed Helpers ========================

def setup_distributed():
    """Initialize distributed process group if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_dist():
    """Check if running in distributed mode."""
    return dist.is_initialized() and dist.get_world_size() > 1


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process():
    return get_rank() == 0


def gather_embeddings_and_ids(features: torch.Tensor, ids: list, dataset_size: int):
    """
    Gather embeddings and ids from all processes.
    DistributedSampler pads the dataset so each rank gets the same number of samples.
    We gather all, then trim to the actual dataset_size to remove duplicates.
    Returns (gathered_features, gathered_ids) on rank 0, (None, None) on others.
    """
    if not is_dist():
        return features, ids

    world_size = get_world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    # --- Gather features (tensor) ---
    local_size = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [int(s.item()) for s in all_sizes]
    max_size = max(all_sizes)

    # Pad features to max_size so all_gather works with uniform shapes
    if features.shape[0] < max_size:
        pad = torch.zeros(max_size - features.shape[0], features.shape[1],
                          dtype=features.dtype, device=device)
        features_padded = torch.cat([features.to(device), pad], dim=0)
    else:
        features_padded = features.to(device)

    gathered_features = [torch.zeros_like(features_padded) for _ in range(world_size)]
    dist.all_gather(gathered_features, features_padded)

    # --- Gather ids (list of python objects) ---
    # Pad ids list to max_size with sentinel
    ids_padded = ids + [None] * (max_size - len(ids))
    gathered_ids_list = [None] * world_size
    dist.all_gather_object(gathered_ids_list, ids_padded)

    if is_main_process():
        # Interleave results from all ranks (DistributedSampler distributes in round-robin)
        # Rank 0 gets indices 0, world_size, 2*world_size, ...
        # Rank 1 gets indices 1, world_size+1, 2*world_size+1, ...
        # We need to restore original order
        total_samples = sum(all_sizes)
        
        # Flatten features and ids from all ranks
        all_feats = []
        all_gathered_ids = []
        for rank_idx in range(world_size):
            n = all_sizes[rank_idx]
            all_feats.append(gathered_features[rank_idx][:n].cpu())
            all_gathered_ids.append(gathered_ids_list[rank_idx][:n])
        
        # Interleave to restore original order
        final_features = []
        final_ids = []
        max_per_rank = max(all_sizes)
        for i in range(max_per_rank):
            for rank_idx in range(world_size):
                if i < all_sizes[rank_idx]:
                    final_features.append(all_feats[rank_idx][i:i+1])
                    final_ids.append(all_gathered_ids[rank_idx][i])
        
        # Trim to actual dataset size (remove padding duplicates)
        final_features = torch.cat(final_features, dim=0)[:dataset_size]
        final_ids = final_ids[:dataset_size]
        
        return final_features, final_ids
    else:
        return None, None

# ======================== Constants ========================

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
            if int(relevance_score) > 0:
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(
        f"Retriever: Average number of relevant documents per query: "
        f"{sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid


def compute_recall_at_k_rewrite(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0
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


# ======================== MMEB Collators ========================

class RzenCandidateDataCollator(SiglipCandidateDataCollator):
    """Extract (text, image_or_None, id) triples from CandidateDataset batches."""

    def __init__(self, image_path_prefix, cand_modal="image,text"):
        self.image_path_prefix = image_path_prefix
        self.cand_modal = cand_modal.split(',') if cand_modal else ["image", "text"]
        self.use_image = "image" in self.cand_modal
        self.use_text = "text" in self.cand_modal

    def __call__(self, batch):
        results = []
        for item in batch:
            candidate_message, did = item
            text_content = ""
            image_path = None
            box = None
            box_op = None

            if candidate_message and len(candidate_message) > 0:
                user_content = candidate_message[0].get('content', [])
                for content_item in user_content:
                    if content_item.get('type') == 'text':
                        text_content = content_item.get('text', '')
                    elif content_item.get('type') == 'image':
                        image_path = content_item.get('image', None)
                        box = content_item.get('box', None)
                        box_op = content_item.get('box_op', BOX_OP) # HACK

            # Remove dataset-specific suffixes
            text_content = text_content.replace("\nSummarize above sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image and sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image in one word: ", "")

            image = None
            if self.use_image and image_path:
                image = Image.open(image_path).convert('RGB')
                image = self.process_image_with_box_op(image, box, box_op)

            if not self.use_text:
                text_content = ""

            # Build instruction text for RzenEmbed
            if image is not None and text_content:
                inst_text = "Represent the given image with related text information: " + text_content
            elif image is not None:
                inst_text = "Represent the given image."
            else:
                inst_text = text_content

            results.append((inst_text, image, did))
        return results


class RzenQueryDataCollator(SiglipCandidateDataCollator):
    """Extract (text, image_or_None, id) triples from QueryDataset batches."""

    def __init__(self, image_path_prefix, query_modal="image,text"):
        self.image_path_prefix = image_path_prefix
        self.cand_modal = query_modal.split(',') if query_modal else ["image", "text"]
        self.use_image = "image" in self.cand_modal
        self.use_text = "text" in self.cand_modal

    def __call__(self, batch):
        results = []
        for item in batch:
            query_message, qid = item
            text_content = ""
            image_path = None
            box = None
            box_op = None

            if query_message and len(query_message) > 0:
                user_content = query_message[0].get('content', [])
                for content_item in user_content:
                    if content_item.get('type') == 'text':
                        text_content = content_item.get('text', '')
                    elif content_item.get('type') == 'image':
                        image_path = content_item.get('image', None)
                        box = content_item.get('box', None)
                        box_op = content_item.get('box_op', BOX_OP) # HACK

            text_content = text_content.replace("\nSummarize above sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image and sentence in one word: ", "")
            text_content = text_content.replace("\nSummarize above image in one word: ", "")

            image = None
            if self.use_image and image_path:
                image = Image.open(image_path).convert('RGB')
                image = self.process_image_with_box_op(image, box, box_op)

            if not self.use_text:
                text_content = ""

            results.append((text_content, image, qid))
        return results


# ======================== Encoding Helper ========================

@torch.no_grad()
def encode_batches(model: RzenEmbed, dataloader: DataLoader, desc="Encoding"):
    """
    Iterate over a dataloader whose collate_fn returns a list of
    (text, image_or_None, id) tuples.  Return (features, ids).
    When running with DistributedSampler, each rank encodes its own shard;
    the caller is responsible for gathering across ranks afterwards.
    """
    all_features = []
    all_ids = []
    disable_tqdm = not is_main_process()  # only show progress on rank 0
    for batch in tqdm(dataloader, desc=desc, disable=disable_tqdm):
        texts = [item[0] for item in batch]
        images = [item[1] for item in batch]
        ids = [item[2] for item in batch]
        all_ids.extend(ids)

        embeddings = model.embed(texts=texts, images=images)
        all_features.append(embeddings.cpu())
    return torch.cat(all_features, dim=0), all_ids


def encode_and_gather(model: RzenEmbed, dataloader: DataLoader, dataset_size: int, desc="Encoding"):
    """
    Encode with the local shard, then gather embeddings + ids from all ranks.
    Returns (features, ids) on rank 0, (None, None) on other ranks.
    In single-GPU mode, simply encodes and returns.
    """
    features, ids = encode_batches(model, dataloader, desc=desc)
    if is_dist():
        features, ids = gather_embeddings_and_ids(features, ids, dataset_size)
    return features, ids


# ======================== MMEB Evaluation ========================

def eval_mmeb(args, model: RzenEmbed):
    if is_main_process():
        print(f"\n===== MMEB Evaluation =====")
        print(f"Query data : {args.query_data_path}")
        print(f"Cand pool  : {args.cand_pool_path}")

    query_dataset = QueryDataset(
        query_data_path=args.query_data_path,
        cand_pool_path=args.query_cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix,
    )
    cand_dataset = CandidateDataset(
        query_data_path=args.query_data_path,
        cand_pool_path=args.cand_pool_path,
        instructions_path=args.instructions_path,
        image_path_prefix=args.image_path_prefix,
    )

    query_collator = RzenQueryDataCollator(
        image_path_prefix=args.image_path_prefix,
        query_modal=args.query_modal,
    )
    cand_collator = RzenCandidateDataCollator(
        image_path_prefix=args.image_path_prefix,
        cand_modal=args.cand_modal,
    )

    batch_size = args.batch_size
    
    # Use DistributedSampler for multi-GPU
    query_sampler = DistributedSampler(query_dataset, shuffle=False) if is_dist() else None
    cand_sampler = DistributedSampler(cand_dataset, shuffle=False) if is_dist() else None
    
    query_dataloader = DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=query_collator, sampler=query_sampler,
    )
    cand_dataloader = DataLoader(
        cand_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=cand_collator, sampler=cand_sampler,
    )

    # Encode and gather embeddings from all GPUs
    query_features, query_ids = encode_and_gather(
        model, query_dataloader, len(query_dataset), desc="Encoding queries"
    )
    cand_features, cand_ids = encode_and_gather(
        model, cand_dataloader, len(cand_dataset), desc="Encoding candidates"
    )

    # Only rank 0 performs evaluation
    if not is_main_process():
        return

    # Top-K retrieval
    index = []
    scores = []
    for i in range(len(query_features)):
        score = query_features[i:i + 1] @ cand_features.T
        topk_score, topk_idx = torch.topk(score, k=50, dim=-1)
        index.append(topk_idx.squeeze().tolist())
        scores.append(topk_score.tolist())

    cand_names = np.array([[unhash_did(cand_ids[item]) for item in row] for row in index])
    query_names = [unhash_qid(item) for item in query_ids]

    # Recall evaluation
    qrel, qid_to_taskid = load_qrel(args.qrels_path)
    k_lists = [1, 5, 10, 50]
    res = {f'recall_{k}': [] for k in k_lists}

    for ind, query_name in enumerate(tqdm(query_names, desc="Computing recall")):
        relevant_docs = qrel[query_name]
        retrieved = cand_names[ind]
        for k in k_lists:
            recall_at_k = compute_recall_at_k_rewrite(relevant_docs, retrieved, k)
            res[f'recall_{k}'].extend(recall_at_k)

    for k in k_lists:
        print(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}']):.4f}")

    # Save results
    save_dir = "./RzenEmbed_eval_results"
    os.makedirs(save_dir, exist_ok=True)
    model_name = args.model_name.split('/')[-1]
    save_name = args.qrels_path.split('/')[-1].replace('_qrels.txt', '')
    with open(f"{save_dir}/{model_name}_mmeb_results.txt", 'a') as f:
        f.write(f"{args.qrels_path}\n")
        for k in k_lists:
            f.write(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}']):.4f}\n")
    print(f"Results written to {save_dir}/{model_name}_mmeb_results.txt")


# ======================== iLIAS Evaluation ========================

class iLiasDatasetForRzen(Dataset):
    """Wraps HuggingFace iLIAS dataset for RzenEmbed evaluation."""

    def __init__(self, hf_split, is_query=True, mode="image"):
        self.data = hf_split
        self.is_query = is_query
        self.mode = mode
        self.key_to_id = {}
        self.id_to_key = {}
        for idx in range(len(self.data)):
            key = self.data[idx]['__key__']
            self.key_to_id[key] = idx
            self.id_to_key[idx] = key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.mode == "text":
            txt = item.get('txt', '')
            return (txt, None, idx)
        image = item['jpg'].convert("RGB")
        bbox_list = item.get('bbox.json', [])
        if bbox_list and len(bbox_list) > 0:
            x, y, w_box, h_box = bbox_list[0]
            img_w, img_h = image.size
            box = [x / img_w, y / img_h, (x + w_box) / img_w, (y + h_box) / img_h]
        else:
            box = [0.0, 0.0, 1.0, 1.0]
        return (image, box, idx)


class IliasCollatorForRzen:
    """Collator converting iLIAS items to (text, image_or_None, id) for RzenEmbed."""

    def __init__(self, mode="query", box_op="crop"):
        self.mode = mode
        self.box_op = box_op

    @staticmethod
    def _crop_image(image, box):
        if box is None:
            return image
        w, h = image.size
        x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            return image.crop((x0, y0, x1, y1))
        return image

    def __call__(self, batch):
        results = []
        for item in batch:
            if self.mode == "query_text":
                raw_text, _, numeric_id = item
                text = f"Find me an image that matches the given caption. {raw_text}" if raw_text else ""
                results.append((text, None, numeric_id))
            elif self.mode == "query":
                image, box, numeric_id = item
                image = self._crop_image(image, box)
                text = "Find the most similar image."
                results.append((text, image, numeric_id))
            else:  # db
                image, box, numeric_id = item
                text = "Represent the given image."
                results.append((text, image, numeric_id))
        return results


def compute_map_at_k(similarity, true_indices, k=50):
    """Compute mAP@K."""
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


def eval_ilias(args, model: RzenEmbed):
    """Evaluate iLIAS dataset with RzenEmbed model."""
    from datasets import load_dataset as hf_load_dataset

    if is_main_process():
        print(f"\n===== iLIAS Evaluation (mode={args.ilias_eval_mode}) =====")

    # Load datasets
    if is_main_process():
        print("Loading database dataset...")
    db_hf = hf_load_dataset(args.ilias_dataset_path, name="core_db", split="core_db")
    db_dataset = iLiasDatasetForRzen(db_hf, is_query=False, mode="image")
    db_key_to_id = db_dataset.key_to_id

    # Build prefix → db ids mapping
    if is_main_process():
        print("Building database prefix index...")
    db_prefix_to_ids = {}
    for db_key, db_idx in db_key_to_id.items():
        prefix = db_key.split('/')[0] if '/' in db_key else db_key
        if prefix not in db_prefix_to_ids:
            db_prefix_to_ids[prefix] = []
        db_prefix_to_ids[prefix].append(db_idx)
    if is_main_process():
        print(f"  Loaded {len(db_dataset)} database items with {len(db_prefix_to_ids)} unique prefixes")

    if is_main_process():
        print("Loading query dataset...")
    if args.ilias_eval_mode == "text":
        query_hf = hf_load_dataset(args.ilias_dataset_path, name="text_queries", split="text_queries")
        query_dataset = iLiasDatasetForRzen(query_hf, is_query=True, mode="text")
        query_collate_mode = "query_text"
    else:
        query_hf = hf_load_dataset(args.ilias_dataset_path, name="img_queries", split="img_queries")
        query_dataset = iLiasDatasetForRzen(query_hf, is_query=True, mode="image")
        query_collate_mode = "query"
    if is_main_process():
        print(f"  Loaded {len(query_dataset)} queries")

    # Build GT mapping
    if is_main_process():
        print("Building ground truth mappings...")
    query_key_to_pos_db_ids = {}
    for q_idx in range(len(query_hf)):
        q_key = query_hf[q_idx]['__key__']
        try:
            prefix = q_key.split('/query/')[0]
        except IndexError:
            continue
        pos_db_ids = db_prefix_to_ids.get(prefix, [])
        query_key_to_pos_db_ids[q_key] = pos_db_ids
    if is_main_process():
        print(f"  Built GT for {len(query_key_to_pos_db_ids)} queries")

    query_data_collator = IliasCollatorForRzen(mode=query_collate_mode)
    db_data_collator = IliasCollatorForRzen(mode="db")

    batch_size = args.batch_size
    
    # Use DistributedSampler for multi-GPU
    query_sampler = DistributedSampler(query_dataset, shuffle=False) if is_dist() else None
    db_sampler = DistributedSampler(db_dataset, shuffle=False) if is_dist() else None
    
    query_dataloader = DataLoader(
        query_dataset, batch_size=batch_size, num_workers=4,
        shuffle=False, collate_fn=query_data_collator, sampler=query_sampler,
    )
    db_dataloader = DataLoader(
        db_dataset, batch_size=batch_size, num_workers=4,
        shuffle=False, collate_fn=db_data_collator, sampler=db_sampler,
    )

    # Encode and gather embeddings from all GPUs
    query_features, query_ids = encode_and_gather(
        model, query_dataloader, len(query_dataset), desc="Encoding queries"
    )
    db_features, db_ids = encode_and_gather(
        model, db_dataloader, len(db_dataset), desc="Encoding database"
    )

    # Only rank 0 performs evaluation
    if not is_main_process():
        return

    # Build GT for mAP
    print("Finalizing ground truth for evaluation...")
    true_indices = []
    for q_numeric_id in tqdm(query_ids, desc="Building GT"):
        q_key = query_dataset.id_to_key[q_numeric_id]
        gt_idx = query_key_to_pos_db_ids.get(q_key, [])
        true_indices.append(gt_idx)

    device = query_features.device
    scores = torch.matmul(query_features.to(device), db_features.to(device).t())
    map_score = compute_map_at_k(scores, true_indices, k=50)

    print(f"\n===== iLIAS Results (Mode: {args.ilias_eval_mode}) =====")
    print(f"Total Queries: {len(query_ids)}")
    print(f"mAP@50: {map_score:.4f}")

    save_dir = "./RzenEmbed_eval_results"
    os.makedirs(save_dir, exist_ok=True)
    model_short = args.model_name.split('/')[-1]
    with open(f"{save_dir}/{model_short}_ilias_results.txt", 'a') as f:
        f.write(f"iLIAS eval_mode={args.ilias_eval_mode}\n")
        f.write(f"mAP@50 = {map_score:.4f}\n")
    print(f"Results written to {save_dir}/{model_short}_ilias_results.txt")


# ======================== Oxford / Paris Evaluation ========================

class OxfordDatasetForRzen(Dataset):
    """Wraps HuggingFace Oxford/Paris dataset for RzenEmbed evaluation."""

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


class OxfordCollatorForRzen:
    """Collator converting Oxford/Paris items to (text, image_or_None, id) for RzenEmbed."""

    def __init__(self, mode="db", box_op="crop"):
        self.mode = mode
        self.box_op = box_op

    @staticmethod
    def _crop_image(image, box):
        if box is None:
            return image
        w, h = image.size
        x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
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
                text = "Find the most similar image."
            else:
                text = "Represent the given image."
            results.append((text, image, item_id))
        return results


def compute_oxford_map(similarity, query_dataset, num_db):
    """Oxford/Paris revisit protocol mAP."""
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


def eval_oxford(args, model: RzenEmbed):
    """Evaluate Oxford5k / Paris6k with RzenEmbed model."""
    from datasets import load_dataset as hf_load_dataset

    dataset_name = getattr(args, 'oxford_dataset_name', 'roxford5k')
    if is_main_process():
        print(f"\n===== Oxford/Paris Evaluation ({dataset_name}) =====")

    dataset_script = args.oxford_dataset_script
    query_hf = hf_load_dataset(dataset_script, name=dataset_name, split="qimlist", trust_remote_code=True)
    db_hf = hf_load_dataset(dataset_script, name=dataset_name, split="imlist", trust_remote_code=True)

    query_dataset = OxfordDatasetForRzen(query_hf, use_bbox=True)
    db_dataset = OxfordDatasetForRzen(db_hf, use_bbox=False)

    box_op = getattr(args, 'oxford_box_op', 'crop')
    query_data_collator = OxfordCollatorForRzen(mode="query", box_op=box_op)
    db_data_collator = OxfordCollatorForRzen(mode="db", box_op=box_op)

    batch_size = args.batch_size
    
    # Use DistributedSampler for multi-GPU
    query_sampler = DistributedSampler(query_dataset, shuffle=False) if is_dist() else None
    db_sampler = DistributedSampler(db_dataset, shuffle=False) if is_dist() else None
    
    query_dataloader = DataLoader(
        query_dataset, batch_size=batch_size, num_workers=8,
        shuffle=False, collate_fn=query_data_collator, sampler=query_sampler,
    )
    db_dataloader = DataLoader(
        db_dataset, batch_size=batch_size, num_workers=8,
        shuffle=False, collate_fn=db_data_collator, sampler=db_sampler,
    )

    # Encode and gather embeddings from all GPUs
    query_features, _ = encode_and_gather(
        model, query_dataloader, len(query_dataset), desc="Encoding queries"
    )
    db_features, _ = encode_and_gather(
        model, db_dataloader, len(db_dataset), desc="Encoding database"
    )

    # Only rank 0 performs evaluation
    if not is_main_process():
        return

    scores = torch.matmul(query_features, db_features.t())
    mAP = compute_oxford_map(scores.cpu(), query_hf, num_db=len(db_hf))

    print(f"\n===== Oxford/Paris Results on {dataset_name} =====")
    print(f"Queries : {len(query_hf)}")
    print(f"Gallery : {len(db_hf)}")
    print(f"mAP     : {mAP:.4f}")

    save_dir = "./RzenEmbed_eval_results"
    os.makedirs(save_dir, exist_ok=True)
    model_short = args.model_name.split('/')[-1]
    with open(f"{save_dir}/{model_short}_oxford_results.txt", 'a') as f:
        f.write(f"dataset={dataset_name}\n")
        f.write(f"mAP = {mAP:.4f}\n")
    print(f"Results written to {save_dir}/{model_short}_oxford_results.txt")


# ======================== Main ========================

def main():
    # Initialize distributed training if using torchrun
    rank, world_size, local_rank = setup_distributed()
    
    parser = argparse.ArgumentParser(
        description='Evaluate RzenEmbed on MMEB / iLIAS / Oxford-Paris datasets'
    )
    parser.add_argument('--eval_task', type=str, default='mmeb',
                        choices=['mmeb', 'ilias', 'oxford'],
                        help='Evaluation task: "mmeb", "ilias", or "oxford"')

    # Model args
    parser.add_argument('--model_name', type=str, default='qihoo360/RzenEmbed',
                        help='Model name or local path for RzenEmbed')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--max_image_tokens', type=int, default=1280,
                        help='Max image tokens for RzenEmbed')
    parser.add_argument('--min_image_tokens', type=int, default=256,
                        help='Min image tokens for RzenEmbed')
    parser.add_argument('--max_length', type=int, default=2000,
                        help='Max sequence length for RzenEmbed')
    parser.add_argument('--image_path_prefix', type=str, default='',
                        help='Prefix path for images')

    # MMEB-specific
    parser.add_argument('--query_data_path', type=str, default=None)
    parser.add_argument('--cand_pool_path', type=str, default=None)
    parser.add_argument('--instructions_path', type=str, default=None)
    parser.add_argument('--qrels_path', type=str, default=None)
    parser.add_argument('--query_cand_pool_path', type=str, default=None)
    parser.add_argument('--query_modal', type=str, default='image,text',
                        help='Modalities for query: "image", "text", or "image,text"')
    parser.add_argument('--cand_modal', type=str, default='image,text',
                        help='Modalities for candidate: "image", "text", or "image,text"')

    # iLIAS-specific
    parser.add_argument('--ilias_dataset_path', type=str, default=None,
                        help='HF dataset path for iLIAS (local dir or hub)')
    parser.add_argument('--ilias_eval_mode', type=str, default='image',
                        choices=['image', 'text'],
                        help='iLIAS eval mode: "image" or "text"')

    # Oxford/Paris-specific
    parser.add_argument('--oxford_dataset_script', type=str,
                        default='dataset/dataset_oxford5k.py',
                        help='Path to Oxford/Paris HF dataset script')
    parser.add_argument('--oxford_dataset_name', type=str, default='roxford5k',
                        choices=['roxford5k', 'rparis6k'],
                        help='Oxford/Paris dataset name')
    parser.add_argument('--oxford_box_op', type=str, default='crop',
                        help='Box operation for Oxford queries: crop/draw/none')

    args = parser.parse_args()

    # Determine device for this process
    if is_dist():
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model once, reuse across tasks
    if is_main_process():
        print(f"Loading RzenEmbed model: {args.model_name}")
        if is_dist():
            print(f"Running with {world_size} GPUs (distributed mode)")
    
    model = RzenEmbed(
        model_path=args.model_name,
        device=device,
        min_image_tokens=args.min_image_tokens,
        max_image_tokens=args.max_image_tokens,
        max_length=args.max_length,
    )

    if args.eval_task == 'mmeb':
        eval_mmeb(args, model)
    elif args.eval_task == 'ilias':
        eval_ilias(args, model)
    elif args.eval_task == 'oxford':
        eval_oxford(args, model)
    
    # Clean up distributed
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
