#! /usr/bin/bash
# parameters: out/in/base-class
# note to add ./checkpoints/ on the input path
# MODEL_ID=qwen2-vl-7b
# RUNNAME=${MODEL_ID}_DAM_ratiog
# STARTCKPT=./checkpoints/Qwen2-VL-7B-Dam

MODEL_ID=qwen2-vl-2b
RUNNAME=${MODEL_ID}_angle
STARTCKPT=$(pwd | awk -F'/usr/' '{print $1}')/dataset/mmeb/Qwen2-VL-2B-Instruct

# stage l1
# bash scripts/lemuir/pretrain.sh ${RUNNAME}_stagel1 ${STARTCKPT} ${MODEL_ID} --use_angle_sim

# stage l1.5
# bash scripts/lemuir/merge_lora.sh ./checkpoints/${RUNNAME}_stagel1-5 ./checkpoints/${RUNNAME}_stagel1 ${STARTCKPT}

# stage l2
# bash scripts/lemuir/finetune.sh ${RUNNAME}_stagel2 ./checkpoints/${RUNNAME}_stagel1-5 ${MODEL_ID} --use_angle_sim

# 评估
# bash scripts/eval/eval_mbeir.sh ./checkpoints/${RUNNAME}_stagel2
# bash scripts/eval/eval_xhs.sh ./checkpoints/${RUNNAME}_stagel2

# --use_angle_sim
# --cos_sim_temp 0.07
# --nocausal_attn


RUNNAME=${MODEL_ID}_temp0.07

# # stage l1
# bash scripts/lemuir/pretrain.sh ${RUNNAME}_stagel1 ${STARTCKPT} ${MODEL_ID} --cos_sim_temp 0.07

# # stage l1.5
# bash scripts/lemuir/merge_lora.sh ./checkpoints/${RUNNAME}_stagel1-5 ./checkpoints/${RUNNAME}_stagel1 ${STARTCKPT}

# # stage l2
# bash scripts/lemuir/finetune.sh ${RUNNAME}_stagel2 ./checkpoints/${RUNNAME}_stagel1-5 ${MODEL_ID} --cos_sim_temp 0.07

# bash scripts/eval/eval_mbeir.sh ./checkpoints/${RUNNAME}_stagel2


RUNNAME=${MODEL_ID}_nocausal

# stage l1
bash scripts/lemuir/pretrain.sh ${RUNNAME}_stagel1 ${STARTCKPT} ${MODEL_ID} --nocausal_attn True

# stage l1.5
bash scripts/lemuir/merge_lora.sh ./checkpoints/${RUNNAME}_stagel1-5 ./checkpoints/${RUNNAME}_stagel1 ${STARTCKPT}

# stage l2
bash scripts/lemuir/finetune.sh ${RUNNAME}_stagel2 ./checkpoints/${RUNNAME}_stagel1-5 ${MODEL_ID} --nocausal_attn

bash scripts/eval/eval_mbeir.sh ./checkpoints/${RUNNAME}_stagel2 --nocausal_attn True