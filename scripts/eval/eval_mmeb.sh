#!/bin/bash
DATASET_NAME=/mnt/tidal-alsh01/dataset/mmeb/mmeb-eval/

IMG_DIR=/mnt/tidal-alsh01/dataset/mmeb/mmeb-eval
ORIGINAL_MODEL_ID=./checkpoints/LEMUIR_Pretrain
MODEL_ID=checkpoints/qwen2_5-vl-7b_LEMUIR_tune
OUTPUT_DIR=./mmeb_eval_results/${MODEL_ID}

IFS=',' read -ra gpu_list <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# A-OKVQA     DocVQA        ImageNet-1K      leaderboard.png       README.md         SUN397
# ChartQA               ImageNet-A       MSCOCO           ObjectNet  RefCOCO           TextVQA            
# CIFAR-100        ImageNet-R              OK-VQA     RefCOCO-Matching              VizWiz
# GQA           images.zip                     ScienceQA         Visual7W           VOC2007
# Country211  HatefulMemes  InfographicsVQA  N24News          Place365   statistics.png    Visual7W-Pointing  

IRET="CIRR MSCOCO_i2t EDIS VisDial VisualNews_i2t VisualNews_t2i MSCOCO_t2i NIGHTS WebQA OVEN FashionIQ Wiki-SS-NQ"
tasks=($IRET)

num_gpus=${#gpu_list[@]}
num_tasks=${#tasks[@]}

group_size=$(( num_tasks / num_gpus ))
remainder=$(( num_tasks % num_gpus ))

start_idx=0

for ((i=0; i<num_gpus; i++)); do
  gpu_id=${gpu_list[$i]}
  curr_group_size=$group_size
  if [[ $i -lt $remainder ]]; then
    curr_group_size=$((group_size + 1))
  fi

  subsets=("${tasks[@]:start_idx:curr_group_size}")
  subset_names="${subsets[*]}"

  echo "GPU $gpu_id 负责 subsets: $subset_names"

  CUDA_VISIBLE_DEVICES=$gpu_id python eval/eval_zeroshot/eval_mmeb.py \
    --dataset_name "$DATASET_NAME" \
    --image_dir "$IMG_DIR" \
    --subset_name $subset_names \
    --image_resolution low \
    --original_model_id "$ORIGINAL_MODEL_ID" \
    --model_id "$MODEL_ID" \
    --batch_size 128 \
    --num_workers 6 \
    --encode_output_path "$OUTPUT_DIR" &

  start_idx=$((start_idx + curr_group_size))
done

wait
echo "所有任务执行完毕。"
