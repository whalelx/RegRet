import json
from transformers import AutoProcessor, AutoTokenizer
import sys 
import os 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
from models.qwen2_vl import Qwen2VLRetForConditionalGeneration
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
import torch
import copy

size = "2"
save_dir = f"./checkpoints/Qwen2-VL-{size}B-Dam"
basemodel = f"/mnt/tidalfs-hssh01/dataset/mmeb/Qwen2-VL-{size}B-Instruct"


base_model = Qwen2VLRetForConditionalGeneration.from_pretrained(basemodel,low_cpu_mem_usage=False,  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
# base_model = Qwen2VLRetForConditionalGeneration.from_pretrained(save_dir,low_cpu_mem_usage=False,  device_map={"": 0},  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

import torch.nn.init as init
for blk, ctx in zip(base_model.visual.blocks, base_model.visual.context_layers):
    ctx.norm1.load_state_dict(copy.deepcopy(blk.norm1.state_dict()))
    ctx.norm2.load_state_dict(copy.deepcopy(blk.norm2.state_dict()))
    ctx.mlp.load_state_dict(copy.deepcopy(blk.mlp.state_dict()))
    ctx.cross_attn.proj.load_state_dict(copy.deepcopy(blk.attn.proj.state_dict()))
    ctx.attn_factor.data.zero_()
    ctx.mlp_factor.data.zero_()
    # ctx.cross_attn.q_proj.load_state_dict(copy.deepcopy(blk.attn.q_proj.state_dict()))
    # ctx.cross_attn.kv_proj.load_state_dict(copy.deepcopy(blk.attn.kv_proj.state_dict()))

    init.xavier_uniform_(ctx.cross_attn.q_proj.weight)
    if ctx.cross_attn.q_proj.bias is not None:
        ctx.cross_attn.q_proj.bias.data.zero_()

    init.xavier_uniform_(ctx.cross_attn.kv_proj.weight)
    if ctx.cross_attn.kv_proj.bias is not None:
        ctx.cross_attn.kv_proj.bias.data.zero_()



# print(base_model.visual.context_layers[0].norm1.weight)
base_model.save_pretrained(save_dir)

processor = AutoProcessor.from_pretrained(basemodel)
processor.save_pretrained(save_dir)  # 这一句会把preprocessor_config.json、chat_template.json等复制过来
tokenizer = AutoTokenizer.from_pretrained(basemodel)
tokenizer.save_pretrained(save_dir)

# python scripts/focal/init_dam_backbone.py