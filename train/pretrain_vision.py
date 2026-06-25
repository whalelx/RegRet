import os
import sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../")
sys.path.append(module_path)
from dataclasses import asdict
import math
from pathlib import Path
from typing import List, Optional
import yaml
import debugpy

from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
from transformers import Trainer, is_datasets_available
import datasets
from transformers.integrations import deepspeed
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, RandomSampler


from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from dataset.datasets_mbeir import LazySupervisedDataset, MbeirLanguageDataset
from dataset.dataset_fgclip import FGCLIPDataset
from dataset.datasets_lemur import XHSDataset
from dataset.datasets_dam import DAMDataset
from dataset.datasets_llavacc3m import LLavaCC3MDataset
# from dataset.datasets_mmeb import MMEBDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils import (
    rank0_print, find_all_linear_names, safe_save_model_for_hf_trainer,
    get_peft_state_maybe_zero_3
)

from trainer import CustomTrainer

def setup_debugpy(local_rank):
    import torch.distributed as dist
    if dist.get_rank() == local_rank:
        print(f"Debugger listening on rank {local_rank}")
        debugpy.listen(("0.0.0.0", 9999))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
    dist.barrier()

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # dumping arguments
    output_dir = getattr(training_args, 'output_dir', None)
    assert output_dir is not None, "output_dir is required"
    args_dir = Path(output_dir) / "arguments"
    args_dir.mkdir(parents=True, exist_ok=True)
    yaml.dump(asdict(model_args), open(args_dir / "model.yaml", "w"))
    yaml.dump(asdict(data_args), open(args_dir / "data.yaml", "w"))
    yaml.dump(asdict(training_args), open(args_dir / "training.yaml", "w"))
    yaml.dump(asdict(lora_args), open(args_dir / "lora.yaml", "w"))

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    device_map = None
    bnb_config = None
    
    # load model, tokenizer, processor
    rank0_print("Loading model, tokenizer, processor...")
    loader = LOADERS[model_args.model_family_id](
        model_hf_path=model_args.model_hf_path,
        model_local_path=model_args.model_local_path,
        compute_dtype=compute_dtype,
        bnb_config=bnb_config,
        use_flash_attn=training_args.use_flash_attn,
        device_map=device_map,
    )
    model, tokenizer, processor = loader.load(pretrain=True)
    tokenizer.model_max_length = training_args.model_max_length

    # Set config parameters from training arguments
    model.config.language_loss_weight = training_args.language_loss_weight
    model.config.use_angle_sim = training_args.use_angle_sim
    model.config.cos_sim_temp = training_args.cos_sim_temp
    model.config.nocausal_attn = training_args.nocausal_attn

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
    for name, param in model.named_parameters():
        param.requires_grad_(False)


    # freeze certain params
    vision_encoder_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]
    # if training_args.train_vision_encoder:
    #     vision_encoder_keys += MODULE_KEYWORDS[model_args.model_family_id]["vision_encoder"]

    rank0_print(f"ctx encoder is freezed... including:")
    for module in vision_encoder_keys:
        rank0_print(f"\t{module}")
        eval(f"model.{module}").requires_grad_(True)

    # vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    # if training_args.train_vision_projector:
    #     rank0_print(f"Vision projector is freezed... including:")
    #     for module in vision_projector_keys:
    #         rank0_print(f"\t{module}")
    #         eval(f"model.{module}").requires_grad_(True)

    # print trainable parameters for inspection
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # param_cnt = sum(p.numel() for p in model.visual.context_layers.parameters() if p.requires_grad) / 1000000
    # rank0_print(f"context encoder extra params: {param_cnt}M")

    # load data
    rank0_print("Loading data...")
    dam_dataset = DAMDataset(
        data_path=data_args.dam_data_path,
        max_length=data_args.dam_max_samples,
        mode = 'none',
    )
    # fgclip_dataset = FGCLIPDataset(
    #     data_path=data_args.fgclip_data_path,
    #     max_length=50000,
    #     use_bbox_ratio=data_args.fgclip_use_bbox_ratio,
    #     # text_truncate_length=350
    # )
    # mbeir_language_dataset = MbeirLanguageDataset(
    #     query_data_path="/mnt/tidal-alsh01/dataset/mmeb/M-BEIR/query/union_train/mbeir_language_train200k.jsonl",
    #     cand_pool_path=data_args.cand_pool_path,
    #     instructions_path=data_args.instructions_path,
    #     image_path_prefix=data_args.image_path_prefix,
    #     tokenizer=tokenizer,
    #     max_length=90000
    # )
    # llavacc3m_dataset = LLavaCC3MDataset(
    #     # image_data_path=data_args.llava_cc3m_data_path,
    #     # json_path=data_args.llava_cc3m_json_path,
    #     max_length=200000
    # )
    # mbeir_dataset = LazySupervisedDataset(
    #     query_data_path=data_args.query_data_path,
    #     cand_pool_path=data_args.cand_pool_path,
    #     instructions_path=data_args.instructions_path,
    #     image_path_prefix=data_args.image_path_prefix,
    #     tokenizer=tokenizer 
    # )
    # mmeb_dataset = MMEBDataset(
    #     data_path=data_args.mmeb_data_path,
    #     type=data_args.mmeb_type,
    #     mode=data_args.mmeb_mode,
    #     max_samples=data_args.mmeb_max_samples
    # )
    # train_dataset = torch.utils.data.ConcatDataset([fgclip_dataset, dam_dataset])
    # train_dataset = torch.utils.data.ConcatDataset([mbeir_language_dataset, llavacc3m_dataset, dam_dataset])
    # train_dataset = torch.utils.data.ConcatDataset([mbeir_dataset, xhs_dataset, dam_dataset, mbeir_language_dataset])
    # train_dataset = torch.utils.data.ConcatDataset([mbeir_dataset, xhs_dataset])
    # train_dataset = torch.utils.data.ConcatDataset([mbeir_dataset, dam_dataset])
    
    eval_dataset = None
    training_args.eval_strategy = "no"

    # data collator
    data_collator = COLLATORS[model_args.model_family_id](
        tokenizer=tokenizer,
        processor=processor,
    )
    training_args.save_strategy = "steps"
    training_args.save_steps = 1000
    training_args.save_total_limit = 1
    
    # training_args.gradient_checkpointing_kwargs = {"use_reentrant": False} # add this one 
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dam_dataset,
    )
    
    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)
    processor.save_pretrained(output_dir)
    copy_json_files(model_args.model_hf_path, output_dir)

def copy_json_files(src_dir, output_dir):
    '''save chat_template.json'''
    import shutil
    source_chat_file = os.path.join(src_dir, "chat_template.json")
    if os.path.exists(source_chat_file):
        target_chat_file = os.path.join(output_dir, "chat_template.json")
        shutil.copy(source_chat_file, target_chat_file)
        rank0_print(f"Copied chat_template.json from {source_chat_file}")
    else:
        rank0_print("Warning: chat_template.json not found in source model")

if __name__ == "__main__":
    train()