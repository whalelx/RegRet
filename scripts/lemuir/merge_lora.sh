# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
#     --original_model_id Qwen/Qwen2-VL-7B-Instruct or Qwen/Qwen2-VL-2B-Instruct \
#     --model_id the_model_path_after_the_first_stage_of_pre-training \
#     --save_path the_path_you_want_to_save

# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
#     --original_model_id /mnt/tidal-alsh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
#     --model_id ./checkpoints/qwen2_5-vl-7b_LEMUIR_Pretrain-Qwen2.5VL \
#     --save_path ./checkpoints/LEMUIR_Pretrain


# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
#     --original_model_id /mnt/tidal-alsh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
#     --model_id ./checkpoints/qwen2_5-vl-7b_LEMUIR_tune_xpretrain \
#     --save_path ./checkpoints/LEMUIR_xPretrain

CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
    --original_model_id /mnt/tidal-alsh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --model_id checkpoints/qwen2_5-vl-7b_MetaQuery_Pretrain-Qwen2.5VL_lora \
    --save_path ./checkpoints/MetaQuery_Pretrain_lora
