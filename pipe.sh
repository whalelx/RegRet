#! /usr/bin/bash
# # parameters: out/in/base-class
# # note to add ./checkpoints/ on the input path
# MODEL_ID=qwen2-vl-7b
# RUNNAME=${MODEL_ID}_DAM_ratiog
# STARTCKPT=./checkpoints/Qwen2-VL-7B-Dam

# # stage 1
# bash scripts/focal/vision_pretrain.sh ${RUNNAME}_stage1 ${STARTCKPT} ${MODEL_ID}
# # stage 2
# bash scripts/focal/train_vllm.sh ${RUNNAME}_stage2 ./checkpoints/${RUNNAME}_stage1 ${MODEL_ID}
# # stage 3
# bash scripts/lemuir/pretrain.sh ${RUNNAME}_stage3 ./checkpoints/${RUNNAME}_stage2 ${MODEL_ID}

# # stage 3.5
# bash scripts/lemuir/merge_lora.sh ./checkpoints/${RUNNAME}_stage3-5 ./checkpoints/${RUNNAME}_stage3

# # stage 4
# bash scripts/lemuir/finetune.sh ${RUNNAME}_stage4 ./checkpoints/${RUNNAME}_stage3-5 ${MODEL_ID}

# # 评估
# bash scripts/eval/eval_mbeir.sh ./checkpoints/${RUNNAME}_stage4
# bash scripts/eval/eval_xhs.sh ./checkpoints/${RUNNAME}_stage4

NODE_RANK=${1:-0}

bash scripts/lemuir/finetune-lemurds.sh ${NODE_RANK}
bash scripts/eval/eval_group_mbeir.sh ${NODE_RANK}