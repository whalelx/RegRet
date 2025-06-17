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
import torch
import transformers
from transformers import Trainer, is_datasets_available
import datasets
from transformers.integrations import deepspeed

from arguments import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from collators import COLLATORS
from dataset.datasets_nli import LazySupervisedDataset
from loaders import LOADERS
from supported_models import MODULE_KEYWORDS
from utils import (
    rank0_print, safe_save_model_for_hf_trainer
)

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
    
    # Enable DeepSpeed if specified
    if getattr(training_args, 'deepspeed', None):
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
    model, tokenizer, processor = loader.load()
    tokenizer.model_max_length = training_args.model_max_length

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Freeze all parameters first
    rank0_print("Freezing all parameters...")
    for param in model.parameters():
        param.requires_grad = False

    vision_projector_keys = MODULE_KEYWORDS[model_args.model_family_id]["vision_projector"]
    if not training_args.train_vision_projector:
        rank0_print(f"Vision projector is freezed... including:")
        for module in vision_projector_keys:
            rank0_print(f"\t{module}")
            eval(f"model.{module}").requires_grad_(False)

    # other components preparation (e.g., image_newline, vision_resampler)
    # we will just freeze these
    if "others" in MODULE_KEYWORDS[model_args.model_family_id]:
        rank0_print(f"Other multimodal component is freezed... including:")
        for other_key in MODULE_KEYWORDS[model_args.model_family_id]["others"]:
            rank0_print(f"\t{other_key}")
            eval(f"model.{other_key}").requires_grad_(False)

    # lora preparation
    llm_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm"]
    llm_heads_keys = MODULE_KEYWORDS[model_args.model_family_id]["llm_heads"]
    if not (lora_args.use_lora or (training_args.train_vision_encoder and lora_args.use_vision_lora)):
        rank0_print("No LoRA enabled...")        
    else:
        named_modules = {n: m for n, m in model.named_modules()}
        lora_modules = []
        full_modules = []

        if training_args.train_vision_encoder and lora_args.use_vision_lora:
            rank0_print("LoRA for vision encoder enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, vision_encoder_keys))
        elif training_args.train_vision_encoder:
            rank0_print("Vision encoder will be fully trained...")
            full_modules.extend(vision_encoder_keys)
        
        if lora_args.use_lora:
            rank0_print("LoRA for LLM enabled...")
            lora_modules.extend(find_all_linear_names(named_modules, llm_keys))
        else:
            rank0_print("LLM will be fully trained...")
            full_modules.extend(llm_keys)
        
        if training_args.train_vision_projector:
            rank0_print("Vision projector will be fully trained...")
            full_modules.extend(vision_projector_keys)

        # Always fully train the embedding head for contrastive learning
        rank0_print("Embedding/Language head will be fully trained...")
        full_modules.extend(llm_heads_keys)
        trainable_modules_keys = MODULE_KEYWORDS[model_args.model_family_id]["trainable_modules"]
        full_modules.extend(trainable_modules_keys)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_modules,
            modules_to_save=full_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
            
        model = get_peft_model(model, lora_config)
        
    # print trainable parameters for inspection
    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(f"\t{name}")

    # load data
    rank0_print("Loading data...")
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
    )
    
    eval_dataset = None
    training_args.eval_strategy = "no"

    # data collator
    data_collator = COLLATORS[model_args.model_family_id](
        tokenizer=tokenizer,
        processor=processor,
    )

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False} # add this one 
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)
    

if __name__ == "__main__":
    train()