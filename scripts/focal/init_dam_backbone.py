import json
from transformers import AutoProcessor, AutoTokenizer
import sys 
import os 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
from models.qwen2_5_vl import Qwen2_5_VLRetForConditionalGeneration
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import torch
import copy

save_dir = "./checkpoints/Qwen2.5-VL-7B-Dam"

base_model = Qwen2_5_VLRetForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",low_cpu_mem_usage=False,  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
# base_model = Qwen2_5_VLRetForConditionalGeneration.from_pretrained(save_dir,low_cpu_mem_usage=False,  device_map={"": 0},  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

import torch.nn.init as init
for blk, ctx in zip(base_model.visual.blocks, base_model.visual.context_layers):
    ctx.norm1.load_state_dict(copy.deepcopy(blk.norm1.state_dict()))
    ctx.norm2.load_state_dict(copy.deepcopy(blk.norm2.state_dict()))
    ctx.mlp.load_state_dict(copy.deepcopy(blk.mlp.state_dict()))
    ctx.cross_attn.proj.load_state_dict(copy.deepcopy(blk.attn.proj.state_dict()))
    # ctx.cross_attn.q_proj.load_state_dict(copy.deepcopy(blk.attn.q_proj.state_dict()))
    # ctx.cross_attn.kv_proj.load_state_dict(copy.deepcopy(blk.attn.kv_proj.state_dict()))

    init.xavier_uniform_(ctx.cross_attn.q_proj.weight)
    if ctx.cross_attn.q_proj.bias is not None:
        ctx.cross_attn.q_proj.bias.data.zero_()

    init.xavier_uniform_(ctx.cross_attn.kv_proj.weight)
    if ctx.cross_attn.kv_proj.bias is not None:
        ctx.cross_attn.kv_proj.bias.data.zero_()

breakpoint()

# print(base_model.visual.context_layers[0].norm1.weight)
base_model.save_pretrained(save_dir)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
processor.save_pretrained(save_dir)  # 这一句会把preprocessor_config.json、chat_template.json等复制过来
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
tokenizer.save_pretrained(save_dir)

# python scripts/focal/init_dam_backbone.py