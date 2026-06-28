# bash scripts/focal/vision_pretrain.sh
bash scripts/focal/train_vllm.sh
bash scripts/lemuir/pretrain.sh
bash scripts/merge_lora.sh

# stage 1
bash scripts/focal/vision_pretrain.sh ${RUNNAME}_stage1 ${STARTCKPT} ${MODEL_ID}
# stage 2
bash scripts/focal/train_vllm.sh ${RUNNAME}_stage2 ./checkpoints/${RUNNAME}_stage1 ${MODEL_ID}
# stage 3
bash scripts/lemuir/pretrain.sh ${RUNNAME}_stage3 ./checkpoints/${RUNNAME}_stage2 ${MODEL_ID}

# stage 3.5
bash scripts/lemuir/merge_lora.sh ./checkpoints/${RUNNAME}_stage3-5 ./checkpoints/${RUNNAME}_stage3

# stage 4
bash scripts/lemuir/finetune.sh ${RUNNAME}_stage4 ./checkpoints/${RUNNAME}_stage3-5 ${MODEL_ID}

# 评估
bash scripts/eval/eval_mbeir.sh ./checkpoints/${RUNNAME}_stage4
bash scripts/eval/eval_xhs.sh ./checkpoints/${RUNNAME}_stage4
