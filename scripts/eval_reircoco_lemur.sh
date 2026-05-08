BOX_OP='concat' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_reircoco.py \
  --json_path /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/REIRCOCO/reircoco_val.json \
  --images_dir /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/REIRCOCO/train2014 \
  --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
  --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs" \
  --batch_size 50 \
  --map_k 50


BOX_OP='crop' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_reircoco.py \
  --json_path /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/REIRCOCO/reircoco_val.json \
  --images_dir /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/REIRCOCO/train2014 \
  --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
  --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs" \
  --batch_size 60 \
  --map_k 50


BOX_OP='crop-lamra' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_reircoco.py \
  --json_path /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/REIRCOCO/reircoco_val.json \
  --images_dir /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/REIRCOCO/train2014 \
  --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
  --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs" \
  --batch_size 60 \
  --map_k 50

BOX_OP='crop-lamra' LAYERWISE=0 CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_reircoco.py \
    --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
    --model_id "checkpoints/LamRA-Ret" \
    --json_path /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/REIRCOCO/reircoco_val.json \
    --images_dir /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/REIRCOCO/train2014 \
    --batch_size 60 \
    --map_k 50






