#! /usr/bin/bash
# parameters: out/in/base-class
# note to add ./checkpoints/ on the input path
MODEL_ID=qwen2_5-vl-7b
RUNNAME=${MODEL_ID}_DAM_ratiog
STARTCKPT=./checkpoints/Qwen2.5-VL-7B-cp

# stage 1
bash scripts/focal/vision_pretrain.sh ${RUNNAME}_stage1 ${STARTCKPT} ${MODEL_ID}

# stage 2
bash scripts/focal/finetune_cp.sh ${RUNNAME}_stage2 ./checkpoints/${RUNNAME}_stage1 ${MODEL_ID}

# 评估
bash scripts/eval/eval_mbeir.sh ./checkpoints/${RUNNAME}_stage4
bash scripts/eval/eval_xhs.sh ./checkpoints/${RUNNAME}_stage4