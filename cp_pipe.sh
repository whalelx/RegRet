#! /usr/bin/bash
# parameters: out/in/base-class
# note to add ./checkpoints/ on the input path
MODEL_ID=qwen2_5-vl-7b
RUNNAME=${MODEL_ID}_DAM_glr
STARTCKPT=./checkpoints/Qwen2.5-VL-7B-cp

# stage 1
bash scripts/focal/vision_pretrain.sh ${RUNNAME}_stage1 ${STARTCKPT} ${MODEL_ID}

# python scripts/focal/merge_dam_lamra.py

# # # stage 2
# bash scripts/focal/finetune_cp.sh ${RUNNAME}_stage2 ./tmp_ckpts/hhh ${MODEL_ID}

# # # 评估

# bash scripts/eval/eval_xhs.sh ./checkpoints/${RUNNAME}_stage2