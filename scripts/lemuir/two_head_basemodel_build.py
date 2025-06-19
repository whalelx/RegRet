import json
from transformers import AutoProcessor
import sys 
import os 
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
from models.qwen2_5_vl import Qwen2_5_VLRetForConditionalGeneration
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
import torch
import copy

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
merge_model = Qwen2_5_VLRetForConditionalGeneration.from_pretrained("./checkpoints/LEMUIR_Pretrain", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

# 主干采用pretrain版本，计算embedding
merge_model.emb_head.norm.load_state_dict(copy.deepcopy(merge_model.model.norm.state_dict()))
merge_model.emb_head.decoder_layers[0].load_state_dict(copy.deepcopy(merge_model.model.layers[-1].state_dict()))

# 然后把qwen原始语言头传给language head
merge_model.model.layers[-1].load_state_dict(copy.deepcopy(base_model.model.layers[-1].state_dict()))
# merge_model.lm_head.load_state_dict(copy.deepcopy(base_model.lm_head.state_dict()))

# 保存合并后的模型
merge_model.save_pretrained("./checkpoints/Higher_Pretrain")

# python scripts/lemuir/two_head_basemodel_build.py