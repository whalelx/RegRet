bash scripts/focal/vision_pretrain.sh
bash scripts/focal/train_vllm.sh
bash scripts/lemuir/pretrain.sh
bash scripts/merge_lora.sh

cd checkpoints/qwen2-vl-2b_DAM_stage3-5
cp /mnt/tidalfs-hssh01/dataset/mmeb/Qwen2-VL-2B-Instruct/chat_template.json .
cp /mnt/tidalfs-hssh01/dataset/mmeb/Qwen2-VL-2B-Instruct/preprocessor_config.json .
cd ../..

bash scripts/lemuir/finetune.sh

