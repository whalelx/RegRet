#!/bin/bash

MODEL_ID=checkpoints/qwen2_5-vl-7b_LEMUIR_tune
ORIGINAL_MODEL_ID=./checkpoints/LEMUIR_Pretrain
DATASET_NAME=/mnt/tidal-alsh01/dataset/mmeb/mmeb-eval/
IMG_DIR=/mnt/tidal-alsh01/dataset/mmeb/mmeb-eval/
OUTPUT_DIR=./mmeb_eval_results
#  N24News ImageNet-A ImageNet-R WebQA GQA Visual7W
python eval/eval_zeroshot/eval_mmeb.py \
    --dataset_name $DATASET_NAME \
    --image_dir $IMG_DIR \
    --subset_name MSCOCO_t2i \
    --image_resolution low \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --batch_size 16 \
    --num_workers 4 \
    --encode_output_path "$OUTPUT_DIR"
