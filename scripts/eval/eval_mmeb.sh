#!/bin/bash
DATASET_NAME=/mnt/tidal-alsh01/dataset/mmeb/mmeb-eval/

IMG_DIR=/mnt/tidal-alsh01/dataset/mmeb/mmeb-eval
ORIGINAL_MODEL_ID=./checkpoints/LEMUIR_Pretrain
MODEL_ID=checkpoints/qwen2_5-vl-7b_LEMUIR_tune
OUTPUT_DIR=./mmeb_eval_results/${MODEL_ID}

# A-OKVQA     DocVQA        ImageNet-1K      leaderboard.png       README.md         SUN397
# ChartQA               ImageNet-A       MSCOCO           ObjectNet  RefCOCO           TextVQA            
# CIFAR-100        ImageNet-R              OK-VQA     RefCOCO-Matching              VizWiz
# GQA           images.zip                     ScienceQA         Visual7W           VOC2007
# Country211  HatefulMemes  InfographicsVQA  N24News          Place365   statistics.png    Visual7W-Pointing  

# CIRR MSCOCO_i2t EDIS VisDial VisualNews_i2t VisualNews_t2i MSCOCO_t2i NIGHTS WebQA OVEN FashionIQ Wiki-SS-NQ
python eval/eval_zeroshot/eval_mmeb.py \
    --dataset_name $DATASET_NAME \
    --image_dir $IMG_DIR \
    --subset_name MSCOCO_t2i \
    --image_resolution low \
    --original_model_id $ORIGINAL_MODEL_ID \
    --model_id $MODEL_ID \
    --batch_size 32 \
    --num_workers 4 \
    --encode_output_path "$OUTPUT_DIR"
