"""
SigLIP-based evaluation script for multimodal retrieval tasks.
This script is based on eval_xhs.py but uses transformers' SigLIP model 
for generating image and text embeddings instead of the custom LamRA model.

The script processes:
- Queries: Text-only inputs processed by SigLIP's text encoder
- Candidates: Image + text inputs processed by both SigLIP encoders and combined

Usage:
python eval_clip.py \
    --query_data_path path/to/query/data \
    --cand_pool_path path/to/candidate/pool \
    --instructions_path path/to/instructions \
    --qrels_path path/to/qrels.txt \
    --query_cand_pool_path path/to/query/candidate/pool \
    --image_path_prefix path/to/images \
    --model_id google/siglip-base-patch16-224
"""
import json
from transformers import SiglipProcessor, SiglipModel, AutoProcessor, AutoModel
try:
    from transformers import Blip2ForImageTextRetrieval,BlipForImageTextRetrieval
except:
    print("fail to import Blip2ForImageTextRetrieval,BlipForImageTextRetrieval")
from transformers import pipeline, CLIPProcessor
import sys 
import os 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
import torch 
import argparse
from dataset.datasets_mbeir import QueryDataset, CandidateDataset
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from accelerate import Accelerator
import accelerate
from tqdm import tqdm
from PIL import Image

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

def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0 # Return 0 if there are no relevant documents

    # Get the set of indices for the top k retrieved documents
    top_k_retrieved_indices_set = set(retrieved_indices[:k])

    # Convert the relevant documents to a set
    relevant_docs_set = set(relevant_docs)

    # Check if there is an intersection between relevant docs and top k retrieved docs
    # If there is, we return 1, indicating successful retrieval; otherwise, we return 0
    result = []
    for x in relevant_docs_set:
        filtered_top_k = top_k_retrieved_indices_set - (relevant_docs_set - {x})
        if x in filtered_top_k:
            result.append(1.0)
        else:
            result.append(0.0)
    return result

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

def normalize_and_tensorize_boxes(bbox, image_width, image_height, feature_size=14):
    x1, y1, x2, y2 = bbox
    newbox = [0] + [i * feature_size for i in bbox]
    boxes_tensor = torch.tensor(newbox, dtype=torch.float32)
    # print(boxes_tensor)
    return boxes_tensor

class SiglipQueryDataCollator:
    def __init__(self, processor, image_path_prefix, query_modal="image,text", image_size=None):
        self.processor = processor
        self.image_path_prefix = image_path_prefix
        self.query_modal = query_modal.split(',') if query_modal else ["image", "text"]
        self.use_image = "image" in self.query_modal
        self.use_text = "text" in self.query_modal
        self.image_size = image_size

    def crop_image_with_box(self, image, box):
        """根据box信息对图片进行crop"""
        if box is None:
            return image
        
        width, height = image.size
        x0 = width * box[0]
        y0 = height * box[1]
        x1 = width * box[2]
        y1 = height * box[3]
        
        # 确保坐标在有效范围内
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        
        # 如果box有效，进行crop
        if x1 > x0 and y1 > y0:
            return image.crop((x0, y0, x1, y1))
        else:
            return image

    def draw_box_on_image(self, image, box):
        """在图片上绘制box"""
        if box is None:
            return image
        
        from PIL import ImageDraw
        width, height = image.size
        x0 = width * box[0]
        y0 = height * box[1]
        x1 = width * box[2]
        y1 = height * box[3]
        
        # 确保坐标在有效范围内
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        
        # 绘制红色边框
        if x1 > x0 and y1 > y0:
            draw = ImageDraw.Draw(image)
            line_thickness = max(2, int(min(width, height) * 0.01))
            draw.rectangle(
                [(x0, y0), (x1, y1)], 
                outline="red", 
                width=line_thickness
            )
        return image

    def process_image_with_box_op(self, image, box, box_op):
        """根据box_op类型处理图片"""
        if box_op == "crop":
            return self.crop_image_with_box(image, box)
        elif box_op == "draw":
            return self.draw_box_on_image(image, box)
        elif box_op == "none" or box_op is None:
            return image
        else:
            # 默认情况，如果不认识的box_op，使用crop
            return self.crop_image_with_box(image, box)
    def process_image_with_box_op2(self, image, box, box_op):
        """根据box_op类型处理图片"""
        if box_op == "crop":
            return self.crop_image_with_box(image, box), None
        elif box_op == "draw":
            return self.draw_box_on_image(image, box), None
        elif box_op == "none" or box_op is None:
            return image
        elif box_op == "concat":
            return image, self.crop_image_with_box(image, box)
        else:
            # 默认情况，如果不认识的box_op，使用crop
            return self.crop_image_with_box(image, box), None
    def __call__(self, batch):
        images = []
        texts = []
        query_ids = []
        boxinfo_tensors = []
        
        for item in batch:
            # item is a tuple: (query_message, qid)
            query_message, qid = item
            query_ids.append(qid)
            
            # 提取文本、图片、box和box_op信息
            text_content = ""
            image_path = None
            box = None
            box_op = None
            
            # 从消息中提取内容
            if query_message and len(query_message) > 0:
                user_content = query_message[0].get('content', [])
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
            
            # 清除prompt
            # text_content = ".".join(text_content.split('.')[1:])
            
            # 根据modal设置决定是否使用文本
            if self.use_text:
                texts.append(text_content)
            else:
                texts.append("")  # 空文本
            
            # 根据modal设置决定是否使用图片
            if self.use_image and image_path:
                image = Image.open(image_path).convert('RGB')
                
                if self.image_size is not None:
                    # 获取原始图片尺寸用于bbox处理
                    image_width, image_height = image.size

                    # 处理boxinfo_tensor
                    if box is not None:
                        if self.image_size == 336:
                            boxinfo_tensor = normalize_and_tensorize_boxes(box, image_width, image_height, feature_size=24)
                        else:
                            boxinfo_tensor = normalize_and_tensorize_boxes(box, image_width, image_height)
                        boxinfo_tensors.append(boxinfo_tensor)
                else:
                    # 根据box_op进行相应处理
                    image = self.process_image_with_box_op(image, box, box_op)
                images.append(image)
            else:
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        # 使用SigLIP processor处理图片和文本
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs.update({'query_ids': query_ids})
        
        # 添加boxinfo_tensor到inputs
        if len(boxinfo_tensors) > 0:
            inputs.update({'boxinfo_tensors': torch.stack(boxinfo_tensors).squeeze(1)})

        return inputs

class SiglipCandidateDataCollator:
    def __init__(self, processor, image_path_prefix, cand_modal="image,text", image_size=None):
        self.processor = processor
        self.image_path_prefix = image_path_prefix
        self.cand_modal = cand_modal.split(',') if cand_modal else ["image", "text"]
        self.use_image = "image" in self.cand_modal
        self.use_text = "text" in self.cand_modal
        self.image_size = image_size

    def crop_image_with_box(self, image, box):
        """根据box信息对图片进行crop"""
        if box is None:
            return image
        
        width, height = image.size
        x0 = width * box[0]
        y0 = height * box[1]
        x1 = width * box[2]
        y1 = height * box[3]
        
        # 确保坐标在有效范围内
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        
        # 如果box有效，进行crop
        if x1 > x0 and y1 > y0:
            return image.crop((x0, y0, x1, y1))
        else:
            return image

    def draw_box_on_image(self, image, box):
        """在图片上绘制box"""
        if box is None:
            return image
        
        from PIL import ImageDraw
        width, height = image.size
        x0 = width * box[0]
        y0 = height * box[1]
        x1 = width * box[2]
        y1 = height * box[3]
        
        # 确保坐标在有效范围内
        x0 = max(0, min(width, x0))
        y0 = max(0, min(height, y0))
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        
        # 绘制红色边框
        if x1 > x0 and y1 > y0:
            draw = ImageDraw.Draw(image)
            line_thickness = max(2, int(min(width, height) * 0.01))
            draw.rectangle(
                [(x0, y0), (x1, y1)], 
                outline="red", 
                width=line_thickness
            )
        return image

    def process_image_with_box_op(self, image, box, box_op):
        """根据box_op类型处理图片"""
        if box_op == "crop":
            return self.crop_image_with_box(image, box)
        elif box_op == "draw":
            return self.draw_box_on_image(image, box)
        elif box_op == "none" or box_op is None:
            return image
        else:
            # 默认情况，如果不认识的box_op，使用crop
            return self.crop_image_with_box(image, box)
    def process_image_with_box_op2(self, image, box, box_op):
        """根据box_op类型处理图片"""
        if box_op == "crop":
            return self.crop_image_with_box(image, box), None
        elif box_op == "draw":
            return self.draw_box_on_image(image, box), None
        elif box_op == "none" or box_op is None:
            return image
        elif box_op == "concat":
            return image, self.crop_image_with_box(image, box)
        else:
            # 默认情况，如果不认识的box_op，使用crop
            return self.crop_image_with_box(image, box), None
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
            
            # 根据modal设置决定是否使用文本
            if self.use_text:
                texts.append(text_content)
            else:
                texts.append("")  # 空文本
            # "\nSummarize above sentence in one word: "
            # 根据modal设置决定是否使用图片
            if self.use_image and image_path:
                image = Image.open(image_path).convert('RGB')
                
                if self.image_size is not None:
                    # 获取原始图片尺寸用于bbox处理
                    image_width, image_height = image.size

                    # 处理boxinfo_tensor
                    if box is not None:
                        if self.image_size == 336:
                            boxinfo_tensor = normalize_and_tensorize_boxes(box, image_width, image_height, feature_size=24)
                        else:
                            boxinfo_tensor = normalize_and_tensorize_boxes(box, image_width, image_height)
                        boxinfo_tensors.append(boxinfo_tensor)
                else:
                    # 根据box_op进行相应处理
                    image = self.process_image_with_box_op(image, box, box_op)
                images.append(image)
            else:
                # 不使用图片或图片路径为空，使用空白图片
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        # 使用SigLIP processor处理图片和文本
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs.update({'candidate_ids': torch.tensor(candidate_ids)})
        
        # 添加boxinfo_tensor到inputs
        if len(boxinfo_tensors) > 0:
            inputs.update({'boxinfo_tensors': torch.stack(boxinfo_tensors).squeeze(1)})
        
        return inputs

def eval(args):
    model_id = args.model_id
    if "blip2" in model_id:
        model = Blip2ForImageTextRetrieval.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    elif "blip" in model_id:
        model = BlipForImageTextRetrieval.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    elif "fg-clip" in model_id:
        from fgclip.model.clip_strc.fgclip import FGCLIPModel
        model = FGCLIPModel.from_pretrained(model_id, ignore_mismatched_sizes=True)
    else:
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    
    model_type = type(model)

    if "EVA-CLIP" in model_id:
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model_type = "EVA-CLIP"
    elif "fg-clip" in model_id:
        processor = CLIPProcessor.from_pretrained(model_id)
        model_type = "fg-clip"
    else:  
        processor = AutoProcessor.from_pretrained(model_id)


    # 解析modal参数
    query_modals = args.query_modal.split(',') if args.query_modal else ["image", "text"]
    cand_modals = args.cand_modal.split(',') if args.cand_modal else ["image", "text"]

    def get_embedding(model, batch, modals, model_type = model_type):
        """根据指定的模态组合生成embedding"""
        if model_type == BlipForImageTextRetrieval:
            batch.update({
                "use_itm_head": False
            })

        if model_type == "EVA-CLIP":
            image_embeds = model.module.encode_image(batch['pixel_values'])
            text_embeds = model.module.encode_text(batch["input_ids"])
        elif model_type == "fg-clip":
            # if "text" in modals:
            #     walk_short_pos=False
            # else:
            #     walk_short_pos=True
            # if (box_info:= getattr(batch, 'boxinfo_tensors', None)) is not None:
            #     image_embeds = model.module.get_image_box_roi_features(batch['pixel_values'], box_info=box_info)
            # else:
            #     image_embeds = model.module.get_image_features(batch['pixel_values'])
            # text_embeds = model.module.get_text_features(batch["input_ids"],walk_short_pos=walk_short_pos)
            # TODO roi align returns poor results
            text_embeds = model.module.get_text_features(batch["input_ids"],walk_short_pos=True) # walk_short_pos 指的是使用短文本
            image_embeds = model.module.get_image_features(batch['pixel_values'])
        else:
            outputs = model(**batch)
            image_embeds = outputs.image_embeds  # [batch_size, hidden_size]
            text_embeds = outputs.text_embeds    # [batch_size, hidden_size]
        
        if model_type == Blip2ForImageTextRetrieval:
            # https://github.com/huggingface/transformers/blob/v4.56.0/src/transformers/models/blip_2/modeling_blip_2.py#L2220
            # blip https://github.com/huggingface/transformers/blob/v4.56.0/src/transformers/models/blip/modeling_blip.py#L126
            #      /usr/local/lib/python3.10/dist-packages/transformers/models/blip/modeling_blip.py
            image_embeds = image_embeds.mean(dim=1)
        
        if "image" in modals and "text" in modals:
            embed = (image_embeds + text_embeds) / 2
        elif "image" in modals:
            embed = image_embeds
        elif "text" in modals:
            embed = text_embeds
        else:
            embed = (image_embeds + text_embeds) / 2
            
        return F.normalize(embed, dim=-1)

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

    query_data_collator = SiglipQueryDataCollator(processor=processor, image_path_prefix=args.image_path_prefix, query_modal=args.query_modal, image_size=args.image_size)
    cand_data_collator = SiglipCandidateDataCollator(processor=processor, image_path_prefix=args.image_path_prefix, cand_modal=args.cand_modal, image_size=args.image_size)
    
    query_dataloader = DataLoader(query_dataset, batch_size=64, num_workers=4, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=64, num_workers=4, shuffle=False, collate_fn=cand_data_collator)

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device 
    is_main_process = accelerator.is_main_process

    model.eval()

    def tensors_to_device(data, device, dtype=model.dtype):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key == 'pixel_values':
                    data[key] = data[key].to(device).to(dtype)
                else:
                    data[key] = data[key].to(device)
        return data 

    query_features = []
    query_ids = []
    candidate_features = []
    candidate_ids = []

    with torch.no_grad():
        query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)
        # 处理查询数据（图片+文本）
        for batch in tqdm(query_dataloader, disable=not is_main_process):
            batch_query_ids = batch.pop('query_ids')
            batch = tensors_to_device(batch, device)
            # 根据query_modal设置生成embedding
            query_embed = get_embedding(model, batch, query_modals)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)

        # 处理候选数据（图片+文本）
        for batch in tqdm(candidate_dataloader, disable=not is_main_process):
            batch_candidate_ids = batch.pop('candidate_ids')
            batch = tensors_to_device(batch, device)
            
            # 根据cand_modal设置生成embedding
            candidate_embed = get_embedding(model, batch, cand_modals)
            
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids.tolist())
            candidate_features.append(candidate_embed)



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
        model_name = args.model_id.split('/')[-1]
        save_name = f"{save_name}_{model_name}"
        # with open(f"{save_dir_name}/{save_name}_query_names.json", 'w') as f:
        #     json.dump(query_names, f, indent=2)
        # with open(f"{save_dir_name}/{save_name}_cand_names.json", 'w') as f:
        #     json.dump(cand_names.tolist(), f, indent=2)
        # with open(f"{save_dir_name}/{save_name}_scores.json", 'w') as f:
        #     json.dump(scores, f, indent=2)
        

        qrel, qid_to_taskid = load_qrel(args.qrels_path)

        k_lists = [1, 5, 10, 50]
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

        model_name = model_id.split('/')[-1]
        with open(f"{save_dir_name}/{model_name}_results.txt", 'a') as f:
            f.write(args.qrels_path + '\n')
            for k in k_lists:
                f.write(f"recall_at_{k} = {sum(res[f'recall_{k}']) / len(res[f'recall_{k}'])}" + '\n')
        print(f"Write output to {save_dir_name}/{model_name}_results.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate retrieval performance using SigLIP model')
    parser.add_argument('--query_data_path', type=str, required=True, help='Path to query data')
    parser.add_argument('--cand_pool_path', type=str, required=True, help='Path to candidate pool data')
    parser.add_argument('--instructions_path', type=str, required=True, help='Path to instructions')
    parser.add_argument('--qrels_path', type=str, required=True, help='Path to qrels file')
    parser.add_argument('--model_id', type=str, default='google/siglip-base-patch16-224', 
                        help='SigLIP model identifier from HuggingFace')
    parser.add_argument('--query_cand_pool_path', type=str, required=True, help='Path to query candidate pool')
    parser.add_argument('--image_path_prefix', type=str, required=True, help='Prefix path for images')
    parser.add_argument('--query_modal', type=str, default='image,text', 
                        help='Modalities for query processing: "image", "text", or "image,text"')
    parser.add_argument('--cand_modal', type=str, default='image,text', 
                        help='Modalities for candidate processing: "image", "text", or "image,text"')
    parser.add_argument('--image_size', type=int, default=None, 
                        help='Image size for resizing (224 or 336)')

    args = parser.parse_args()
    eval(args)