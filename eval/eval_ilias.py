import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from datasets import load_dataset

# 假设你的自定义模块路径
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)

from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from loaders.processor import LemuirProcessor
from collators.qwen2_vision_process import process_vision_info_with_focal

BOX_OP = os.environ.get("BOX_OP", "crop")

def add_embed_token(tokenizer, model, emb_token="<emb>"):
    num_new_tokens = tokenizer.add_tokens([emb_token])
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.config.emb_token_ids = tokenizer.convert_tokens_to_ids([emb_token])

# ========== mAP@K 计算函数 ==========
def compute_map_at_k(similarity, true_indices, k=50):
    """
    计算 mAP@K
    
    Args:
        similarity: shape [num_queries, num_candidates] 的相似度矩阵
        true_indices: 一个列表，长度为 num_queries。
                      每个元素是一个 list 或 tensor，包含该 query 对应的正确 candidate 下标。
        k: 计算 Top K (默认 50)
        
    Returns:
        mAP@K 分数
    """
    if not isinstance(similarity, torch.Tensor):
        similarity = torch.tensor(similarity)
    device = similarity.device
    num_queries = similarity.shape[0]

    # 获取所有 Query 的前 K 个预测索引
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

        # 判断命中情况
        if hasattr(torch, 'isin'): 
            hits = torch.isin(pred_indices, gt_indices)
        else:
            hits = (pred_indices.unsqueeze(1) == gt_indices.unsqueeze(0)).any(dim=1)

        if hits.sum() == 0:
            average_precisions.append(0.0)
            continue

        # 计算 AP@K
        hits = hits.float()
        cumsum_hits = torch.cumsum(hits, dim=0)
        ranks = torch.arange(1, k + 1, device=device, dtype=torch.float)
        precision_at_i = cumsum_hits / ranks
        
        # 使用标准定义: 除以总的正确答案数量
        score = (precision_at_i * hits).sum() / num_gt
        average_precisions.append(score.item())

    if not average_precisions:
        return 0.0
        
    mAP = sum(average_precisions) / len(average_precisions)
    return mAP
# ==========================================

class iIiasDataset(Dataset):
    def __init__(self, hf_split, is_query=True, mode="image"):
        self.data = hf_split
        self.is_query = is_query
        self.mode = mode # "image" or "text"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        key = item['__key__']
        
        # --- 文本模式逻辑 ---
        if self.mode == "text":
            # 获取文本内容
            txt_content = item.get('txt', '')
            return {
                "image": None,
                "box": None,
                "text": txt_content,
                "key": key
            }
        
        # --- 图像模式逻辑 (默认) ---
        image = item['jpg'].convert("RGB")
        
        bbox_list = item.get('bbox.json', [])
        if bbox_list and len(bbox_list) > 0:
            x, y, w_box, h_box = bbox_list[0]
            img_w, img_h = image.size
            box = [x / img_w, y / img_h, (x + w_box) / img_w, (y + h_box) / img_h]
        else:
            box = [0.0, 0.0, 1.0, 1.0]

        return {
            "image": image,
            "box": box,
            "text": None,
            "key": key
        }

def collate_fn(batch, processor, tokenizer, mode="query"):
    """
    mode: "query" (image), "query_text", "db"
    """
    messages_list = []
    keys = []
    
    # 判断是否为文本查询模式
    is_text_mode = (mode == "query_text")
    
    for item in batch:
        keys.append(item['key'])
        
        if is_text_mode:
            # --- 文本查询 Prompt ---
            # 这里的 text 是 dataset 返回的 txt 字段
            raw_text = item['text'] if item['text'] else ""
            text_prompt = f"Find me an image that matches the given caption. {raw_text}\nSummarize above sentence in one word: "  if mode == "query_text" else "Summarize the image in one word: "
            
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt}
                    ]
                },
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]}
            ]
        else:
            text_prompt = "Find the most similar image.\nSummarize the image and text in one word: " if mode == "query" else "Summarize the image in one word: "
            
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item['image'], "box": item['box'], "box_op": BOX_OP},
                        {"type": "text", "text": text_prompt}
                    ]
                },
                {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]}
            ]
        
        messages_list.append(msgs)


    
    image_inputs, id_dict = process_vision_info_with_focal(messages_list, box_op=BOX_OP)
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_list]
    
    replace_two_imgs = set(id_dict.multi_img_texts) if hasattr(id_dict, 'multi_img_texts') else set()
    inputs, crop_or_concat_img_inputs = processor(
        text=texts, images=image_inputs, videos=None, 
        padding=True, return_tensors="pt", id_dict=id_dict,
        replace_two_imgs=replace_two_imgs
    )

    input_ids = inputs['input_ids']
    labels = input_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    
    model_inputs = {
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs.get('attention_mask'),
        "pixel_values": None, 
        "labels": labels,
    }
    
    # 图像模式才添加视觉相关的输入
    if not is_text_mode:
        model_inputs["image_grid_thw"] = inputs.get('image_grid_thw')
        model_inputs.update(crop_or_concat_img_inputs)
        
    return model_inputs, id_dict, keys

def get_embeddings(dataloader, model, accelerator):
    model.eval()
    all_embeds = []
    
    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        model_inputs, id_dict, keys = batch
        model_inputs = {k: v.to(accelerator.device) for k, v in model_inputs.items() 
                        if v is not None and isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            # 模型需要支持 inference=True 且输入中无图像时的处理逻辑
            embeds = model(**model_inputs, id_dict=id_dict, inference=True)
            embeds = F.normalize(embeds, dim=-1)
            
            gathered_embeds = accelerator.gather_for_metrics(embeds)
            all_embeds.append(gathered_embeds.cpu())
            
    return torch.cat(all_embeds, dim=0)

def run_eval(args):
    accelerator = Accelerator()
    
    # 1. 加载模型与处理器
    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    processor = LemuirProcessor.from_pretrained(args.original_model_id or args.model_id)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'

    add_embed_token(processor.tokenizer, model)
    
    # 2. 加载数据集
    db_dataset = iIiasDataset(load_dataset(args.dataset_path, name="core_db", split="core_db"), is_query=False, mode="image")
    
    # 根据参数选择 Query Split
    if args.eval_mode == "text":
        accelerator.print("Evaluating in Text Query Mode...")
        query_hf_split = load_dataset(args.dataset_path, name="text_queries", split="text_queries")
        query_dataset = iIiasDataset(query_hf_split, is_query=True, mode="text")
        query_collate_mode = "query_text"
    else:
        accelerator.print("Evaluating in Image Query Mode...")
        query_hf_split = load_dataset(args.dataset_path, name="img_queries", split="img_queries")
        query_dataset = iIiasDataset(query_hf_split, is_query=True, mode="image")
        query_collate_mode = "query"

    db_loader = DataLoader(db_dataset, batch_size=args.batch_size, 
                           collate_fn=lambda b: collate_fn(b, processor, processor.tokenizer, "db"))
    query_loader = DataLoader(query_dataset, batch_size=args.batch_size, 
                              collate_fn=lambda b: collate_fn(b, processor, processor.tokenizer, query_collate_mode))

    model, db_loader, query_loader = accelerator.prepare(model, db_loader, query_loader)

    # 3. 提取特征
    db_embeds = get_embeddings(db_loader, model, accelerator)
    query_embeds = get_embeddings(query_loader, model, accelerator)

    if accelerator.is_main_process:
        # 4. 重建 Keys
        print("Reconstructing keys from dataset...")
        db_keys = [item['key'] for item in tqdm(db_dataset, desc="Loading DB Keys")]
        query_keys = [item['key'] for item in tqdm(query_dataset, desc="Loading Query Keys")]
        
        # 5. 计算 Similarity Matrix
        db_embeds = db_embeds.to(accelerator.device)
        query_embeds = query_embeds.to(accelerator.device)
        scores = torch.matmul(query_embeds, db_embeds.t())
        
        # 6. 构建 Ground Truth Indices
        true_indices = []
        
        print("Building ground truth indices...")
        for q_key in tqdm(query_keys, desc="Building GT"):
            # 逻辑：bold_bimp_000/query/... -> 对应的正例前缀是 bold_bimp_000/pos/
            # 注意：text_queries 的 key 格式应与 img_queries 保持一致才能匹配到 db
            try:
                prefix = q_key.split('/query/')[0]
                pos_prefix = f"{prefix}/pos/"
            except IndexError:
                true_indices.append([])
                continue
            
            gt_idx = [idx for idx, k in enumerate(db_keys) if k.startswith(pos_prefix)]
            true_indices.append(gt_idx)
        
        # 7. 计算 mAP@50
        print("Computing mAP@50...")
        map_score = compute_map_at_k(scores, true_indices, k=50)
        
        print(f"\nFinal Results (Mode: {args.eval_mode}):")
        print(f"Total Queries: {len(query_keys)}")
        print(f"mAP@50: {map_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help="HF dataset path or local directory")
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--original_model_id', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_mode', type=str, default="image", choices=["image", "text"], 
                        help="Evaluation mode: 'image' for img_queries, 'text' for text_queries")
    args = parser.parse_args()
    run_eval(args)
