#! /usr/bin/bash
# parameters: out/in/base-class
# note to add ./checkpoints/ on the input path
MODEL_ID=qwen2-vl-7b
RUNNAME=${MODEL_ID}_Lemur_Pretrain_585ktxt_nocausal
STARTCKPT=./checkpoints/Qwen2-VL-7B-Dam


# cd checkpoints
# mv qwen2-vl-7b_Lemur_Pretrain_585ktxt_nocausal qwen2-vl-7b_Lemur_Pretrain_585ktxt
# cd ..

# # stage 3.5
bash scripts/lemuir/merge_lora.sh ./tmp_ckpts/${RUNNAME}_stage1-5 ./checkpoints/${RUNNAME} /mnt/tidal-alsh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/

python scripts/focal/merge_dam_lamra.py
rm -rf ./tmp_ckpts/${RUNNAME}_stage1-5
# Lemur_7B_Pretrain_585ktxt_nocausal_enc
# stage 4
# bash scripts/lemuir/finetune-lemurds.sh 0
# ${RUNNAME}_stage2 "checkpoints/Lemur_7B_Pretrain_585ktxt_enc" ${MODEL_ID} --nocausal_attn

# 评估
# bash scripts/eval/eval_mbeir.sh ${MODEL_ID}_Lemur_585ktxt_tune_stage2
# bash scripts/eval/eval_xhs.sh ./checkpoints/${RUNNAME}_stage2
