import os
from transformers import AutoProcessor
import sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
import torch
import argparse
import torch.nn.functional as F
from accelerate import Accelerator
import accelerate
from peft import PeftModel
import shutil



def add_embed_token(tokenizer, model, emb_token="<emb>"):
    emb_tokens = [emb_token]
    num_new_tokens = tokenizer.add_tokens(emb_tokens)
    if num_new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    emb_token_ids = tokenizer.convert_tokens_to_ids(emb_tokens)
    model.config.emb_token_ids = emb_token_ids

def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id
    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        original_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(original_model_id)
    tokenizer = processor.tokenizer

    # Add the embed token to match the training setup
    add_embed_token(tokenizer, model)

    lora_model = PeftModel.from_pretrained(model, model_id)
    merged_model = lora_model.merge_and_unload()

    # merged_model.save_pretrained
    merged_model.save_pretrained(args.save_path)
    processor.save_pretrained(args.save_path)

    # copy the chat_template.json file
    source_chat_file = os.path.join(args.original_model_id, "chat_template.json")
    target_chat_file = os.path.join(args.save_path, "chat_template.json")
    shutil.copy(source_chat_file, target_chat_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--save_path', type=str)

    args = parser.parse_args()
    eval(args)