# BOX_OP='crop' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_ilias.py \
#     --dataset_path /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/ilias \
#     --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
#     --batch_size 60 \
#     --eval_mode text \
#     --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs"




# BOX_OP='crop-lamra' LAYERWISE=0 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_ilias.py \
#     --dataset_path /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/ilias \
#     --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
#     --batch_size 60 \
#     --eval_mode text \
#     --model_id "checkpoints/LamRA-Ret" 


BOX_OP='crop' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_ilias.py \
    --dataset_path /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/ilias \
    --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
    --batch_size 60 \
    --eval_mode image \
    --model_id "checkpoints/Lemur_8B_zeroshot-enc"