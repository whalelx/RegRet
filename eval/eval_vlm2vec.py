import json
import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

# Import MMEBModel and collators from eval.py
from transformers import HfArgumentParser, AutoProcessor,AutoConfig
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
import torch 
import argparse
from dataset.datasets_mbeir import QueryDataset, CandidateDataset

from vlm2vec.src.arguments import ModelArguments, DataArguments, TrainingArguments
from vlm2vec.src.model import MMEBModel
from vlm2vec.src.collator import EvalCollator
from vlm2vec.src.model_utils import get_backbone_name
from vlm2vec.src.model_utils import load_processor, QWEN2_VL, vlm_image_tokens

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
    def __init__(self, processor, image_path_prefix, cand_modal="image,text", image_size=None):
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
    def __init__(self, processor, image_path_prefix, cand_modal="image,text", image_size=None):
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

def eval_mmeb(args):
    print(f"Starting MMEB evaluation with model: {args.model_name}")
    print(f"Model backbone: {args.model_backbone}")
    assert args.model_backbone == "mllama"
    print(f"Batch size: {getattr(args, 'batch_size', 64)}")
    
    # Setup arguments similar to eval.py
    model_args = ModelArguments(model_name=args.model_name)
    data_args = DataArguments()
    training_args = TrainingArguments()
    
    # Configure arguments based on input args
    model_args.model_name = args.model_name
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    model_args.model_backbone = model_backbone
    model_args.normalize = args.normalize
    model_args.pooling = args.pooling
    model_args.num_crops = getattr(args, 'num_crops', 4)  # Default value for llava_next
    
    data_args.image_dir = args.image_path_prefix
    data_args.max_len = 1024
    
    # training_args.device = "cuda"
    training_args.per_device_eval_batch_size = getattr(args, 'batch_size', 64)
    training_args.dataloader_num_workers = 0
    
    from transformers import LlavaNextProcessor
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        trust_remote_code=True
    )

    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)
    
    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )
    
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

    query_data_collator = MME5QueryDataCollator(processor=processor, image_path_prefix=args.image_path_prefix, cand_modal=args.query_modal )
    cand_data_collator = MME5CandidateDataCollator(processor=processor, image_path_prefix=args.image_path_prefix, cand_modal=args.cand_modal )
    
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
    query_features = []
    query_ids = []
    candidate_features = []
    candidate_ids = []
    
    with torch.no_grad():
        # Encode queries
        for batch in tqdm(query_dataloader, desc="Encoding queries"):
            # Extract IDs first
            batch_ids = batch.get('qiddid', [])
            if isinstance(batch_ids, torch.Tensor):
                query_ids.extend(batch_ids.cpu().numpy().tolist())
            else:
                query_ids.extend(batch_ids)
            
            # Remove ID field and move to device
            batch = {key: value.to(device) for key, value in batch.items() if key != 'qiddid'}
            
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                output = model(qry=batch)
                query_features.append(output["qry_reps"])
        
        # Encode candidates
        for batch in tqdm(candidate_dataloader, desc="Encoding candidates"):
            # Extract IDs first
            batch_ids = batch.get('qiddid', [])
            if isinstance(batch_ids, torch.Tensor):
                candidate_ids.extend(batch_ids.cpu().numpy().tolist())
            else:
                candidate_ids.extend(batch_ids)
            
            # Remove ID field and move to device
            batch = {key: value.to(device) for key, value in batch.items() if key != 'qiddid'}
            
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                output = model(tgt=batch)
                candidate_features.append(output["tgt_reps"])

    # Concatenate features
    query_features = torch.cat(query_features, dim=0)
    candidate_features = torch.cat(candidate_features, dim=0)
    
    if is_main_process:
        # Adjust the order according to ids 
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
    parser = argparse.ArgumentParser(description='Evaluate retrieval performance using MMEB model with phi3_v')
    parser.add_argument('--query_data_path', type=str, required=True, help='Path to query data')
    parser.add_argument('--cand_pool_path', type=str, required=True, help='Path to candidate pool data')
    parser.add_argument('--instructions_path', type=str, default=None, help='Path to instructions')
    parser.add_argument('--qrels_path', type=str, required=True, help='Path to qrels file')
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path')
    parser.add_argument('--model_backbone', type=str, default='mllama', help='Model backbone (phi35v)')
    parser.add_argument('--query_cand_pool_path', type=str, required=True, help='Path to query candidate pool')
    parser.add_argument('--image_path_prefix', type=str, required=True, help='Prefix path for images')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num_crops', type=int, default=4, help='Number of crops for phi3_v processor')
    parser.add_argument("--pooling", type=str, default="last", help="Pooling method for MMEB model")
    parser.add_argument("--normalize", type=bool, default=True, help="Whether to normalize features")
    parser.add_argument('--query_modal', type=str, default='image,text', 
                        help='Modalities for query processing: "image", "text", or "image,text"')
    parser.add_argument('--cand_modal', type=str, default='image,text', 
                        help='Modalities for candidate processing: "image", "text", or "image,text"')
    args = parser.parse_args()
    eval_mmeb(args)