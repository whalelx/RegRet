'''
1. 使用lemuir-master的脚本，把nodam模型（用lora训练的）merge成./checkpoints/tempckpt.参考脚本：
    CUDA_VISIBLE_DEVICES='0' python merge_lora/merge.py \
        --original_model_id ./checkpoints/LEMUIR_Pretrain \
        --model_id ./checkpoints/qwen2_5-vl-7b_LEMUIR_tune_nodam \
        --save_path ./checkpoints/tempckpt

2. 使用当前脚本合并权重
'''
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

save_dir = "./tmp_ckpts/dam_pretrain+lamrallm"

base_model = Qwen2_5_VLRetForConditionalGeneration.from_pretrained("./checkpoints/qwen2_5-vl-7b_DAM_pretrain_vision",low_cpu_mem_usage=False,  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
lamra_ret = Qwen2_5_VLRetForConditionalGeneration.from_pretrained("./checkpoints/tempckpt",low_cpu_mem_usage=False,  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

lamra_ret.visual.context_layers.load_state_dict(copy.deepcopy(base_model.visual.context_layers.state_dict()))

# for blk, ctx in zip(base_model.visual.context_layers, lamra_ret.visual.context_layers):
#     ctx.norm1.load_state_dict(copy.deepcopy(blk.norm1.state_dict()))
#     ctx.norm2.load_state_dict(copy.deepcopy(blk.norm2.state_dict()))
#     ctx.mlp.load_state_dict(copy.deepcopy(blk.mlp.state_dict()))
#     ctx.cross_attn.load_state_dict(copy.deepcopy(blk.cross_attn.state_dict()))
#     ctx.attn_factor.data.copy_(blk.attn_factor.data)
#     ctx.mlp_factor.data.copy_(blk.mlp_factor.data)
#     # ctx.cross_attn.q_proj.load_state_dict(copy.deepcopy(blk.attn.q_proj.state_dict()))
#     # ctx.cross_attn.kv_proj.load_state_dict(copy.deepcopy(blk.attn.kv_proj.state_dict()))

print("@attn_factor: ", lamra_ret.visual.context_layers[0].attn_factor)
print("@gt attn_factor: ", base_model.visual.context_layers[0].attn_factor)

lamra_ret.save_pretrained(save_dir)

base_model = Qwen2_5_VLRetForConditionalGeneration.from_pretrained("./tmp_ckpts/dam_pretrain+lamrallm",low_cpu_mem_usage=False,  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
breakpoint()

processor = AutoProcessor.from_pretrained("./checkpoints/LEMUIR_Pretrain")
processor.save_pretrained(save_dir)  # 这一句会把preprocessor_config.json、chat_template.json等复制过来
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/LEMUIR_Pretrain")
tokenizer.save_pretrained(save_dir)

# python scripts/focal/merge_dam_lamra.py