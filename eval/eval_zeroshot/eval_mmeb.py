import json
import sys
import os
import argparse

current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)

from transformers import AutoProcessor
from models.qwen2_5_vl import Qwen2_5_VLRetForConditionalGeneration
from dataset.datasets_mmeb import MMEBEvalDataset
from collators.mmeb_collator import MMEBEvalCollator

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F 
from tqdm import tqdm
import numpy as np
import pickle
from datasets import load_dataset

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def get_pred(qry_t, tgt_t, normalization=False):
    """
    Use L2 norms.
    """
    if normalization:
        qry_t_norm = np.linalg.norm(qry_t)
        tgt_t_norms = np.linalg.norm(tgt_t, axis=1)
        scores = np.dot(tgt_t, qry_t) / (tgt_t_norms * qry_t_norm)
    else:
        scores = np.dot(tgt_t, qry_t)
    pred = np.argmax(scores)
    return scores, pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="TIGER-Lab/MMEB-eval")
    parser.add_argument('--subset_name', type=str, nargs='+', help="List of subset names to evaluate")
    parser.add_argument('--image_dir', type=str, help="Path to the image directory")
    parser.add_argument('--image_resolution', type=str, default="low", choices=['low', 'mid', 'high'])
    parser.add_argument('--original_model_id', type=str, help="Original model ID for processor")
    parser.add_argument('--model_id', type=str, help="Fine-tuned model ID")
    parser.add_argument('--model_max_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--encode_output_path', type=str, help="Output path for encoded features")

    args = parser.parse_args()
    os.makedirs(args.encode_output_path, exist_ok=True)

    # Load model
    model = Qwen2_5_VLRetForConditionalGeneration.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", 
        low_cpu_mem_usage=True, 
    )

    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(args.original_model_id)

    tokenizer = processor.tokenizer 
    tokenizer.padding_side = 'left'
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, dtype=torch.bfloat16)

    data_args = {
        "dataset_name": args.dataset_name,
        "image_resolution": args.image_resolution,
        "image_dir": args.image_dir
    }

    eval_collator = MMEBEvalCollator(tokenizer=tokenizer, processor=processor)

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(args.subset_name):
        score_path = os.path.join(args.encode_output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
            except Exception as e:
                pass

        print(f"\033[91m{idx+1}/{len(args.subset_name)}: Processing {subset} now!\033[0m")
        encode_qry_path = os.path.join(args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(args.encode_output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            print(f"Found existing encoded features for {subset}, skipping encoding")
            continue
        else:
            eval_qry_dataset = MMEBEvalDataset(
                data_args=data_args,
                subset=subset,
                text_field="qry_text",
                img_path_field="qry_img_path",
            )
            eval_tgt_dataset = MMEBEvalDataset(
                data_args=data_args,
                subset=subset,
                text_field="tgt_text",
                img_path_field="tgt_img_path",
            )

            eval_qry_loader = DataLoader(
                eval_qry_dataset,
                batch_size=args.batch_size,
                collate_fn=eval_collator,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
            )
            eval_tgt_loader = DataLoader(
                eval_tgt_dataset,
                batch_size=args.batch_size,
                collate_fn=eval_collator,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
            )


            encoded_tensor = []
            with torch.no_grad():
                for batch in tqdm(eval_qry_loader, desc="Encode query"):
                    batch = batch_to_device(batch, device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        embeddings = model(**batch, inference=True)
                        embeddings = F.normalize(embeddings, dim=-1)
                        encoded_tensor.append(embeddings.cpu().detach().float().numpy())
            encoded_tensor = np.concatenate(encoded_tensor)
            with open(encode_qry_path, 'wb') as f:
                pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

            # Encode targets
            encoded_tensor = []
            with torch.no_grad():
                for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                    batch = batch_to_device(batch, device)
                    with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                        embeddings = model(**batch, inference=True)
                        embeddings = F.normalize(embeddings, dim=-1)
                        encoded_tensor.append(embeddings.cpu().detach().float().numpy())
            encoded_tensor = np.concatenate(encoded_tensor)
            with open(encode_tgt_path, 'wb') as f:
                pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    # Calculate scores for each subset
    for subset in tqdm(args.subset_name, desc="calculate score"):
        encode_qry_path = os.path.join(args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(args.encode_output_path, f"{subset}_tgt")
        
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)
            
        qry_dict, tgt_dict = {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            tgt_dict[(text, img_path)] = tgt_t

        eval_data = load_dataset(
            args.dataset_name,
            subset,
            split="test",
        )
        
        n_correct = 0
        all_pred = []
        for row in eval_data:
            qry_t = qry_dict[(row["qry_text"], row["qry_img_path"])]  # (dim,)
            tgt_t, all_candidates = [], []
            for tt in zip(row["tgt_text"], row["tgt_img_path"]):
                tgt_t.append(tgt_dict[tt])
                all_candidates.append(tt)
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            scores, pred = get_pred(qry_t, tgt_t, normalization=False)
            if pred == 0:
                n_correct += 1
            all_pred.append(all_candidates[pred])
            
        with open(os.path.join(args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")
                
        score_path = os.path.join(args.encode_output_path, f"{subset}_score.json")
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {"acc": n_correct/len(eval_data), "num_correct": n_correct, "num_pred": len(eval_data)}
            json.dump(score_dict, f, indent=4)
        print(f"\033[91m{subset} accuracy: {n_correct/len(eval_data)}\033[0m")


if __name__ == "__main__":
    main()
