# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
#     --original_model_id Qwen/Qwen2-VL-7B-Instruct or Qwen/Qwen2-VL-2B-Instruct \
#     --model_id the_model_path_after_the_first_stage_of_pre-training \
#     --save_path the_path_you_want_to_save
original_model_id=${3:-"/mnt/tidal-alsh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"}
CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 merge_lora/merge.py \
    --original_model_id checkpoints/Qwen2-VL-7B-Dam \
    --model_id ./checkpoints/qwen2-vl-7b_DAM_stage3 \
    --save_path ./checkpoints/qwen2-vl-7b_DAM_stage3-5
