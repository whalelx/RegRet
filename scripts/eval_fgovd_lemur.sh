# BOX_OP='crop' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_fgovd.py \
#     --jsonl_path "/mnt/tidalfs-hssh01/usr/liangxun/ICLR26/FG-CLIP/data/fgovd/e_attributes_llava.jsonl" \
#     --image_root "/mnt/tidalfs-hssh01/dataset/mmeb/COCO" \
#     --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
#     --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs"

BOX_OP='crop' LAYERWISE=1 CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_fgovd.py \
    --jsonl_path "/mnt/tidalfs-hssh01/usr/liangxun/ICLR26/FG-CLIP/data/fgovd/e_attributes_llava.jsonl" \
    --image_root "/mnt/tidalfs-hssh01/dataset/mmeb/COCO" \
    --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
    --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_final_tune_stage2-largebs"

BOX_OP='crop' LAYERWISE=1 CUDA_VISIBLE_DEVICES='2' accelerate launch --main_process_port 29501 --multi_gpu eval/eval_fgovd.py \
    --jsonl_path "/mnt/tidalfs-hssh01/usr/liangxun/ICLR26/FG-CLIP/data/fgovd/m_attributes_llava.jsonl" \
    --image_root "/mnt/tidalfs-hssh01/dataset/mmeb/COCO" \
    --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
    --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs"



BOX_OP='crop' LAYERWISE=1 CUDA_VISIBLE_DEVICES='3' accelerate launch --main_process_port 29502 --multi_gpu eval/eval_fgovd.py \
    --jsonl_path "/mnt/tidalfs-hssh01/usr/liangxun/ICLR26/FG-CLIP/data/fgovd/h_attributes_llava.jsonl" \
    --image_root "/mnt/tidalfs-hssh01/dataset/mmeb/COCO" \
    --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
    --model_id "checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs"


BOX_OP='crop-lamra' LAYERWISE=0 CUDA_VISIBLE_DEVICES='4' accelerate launch --main_process_port 29503 --multi_gpu eval/eval_fgovd.py \
    --jsonl_path "/mnt/tidalfs-hssh01/usr/liangxun/ICLR26/FG-CLIP/data/fgovd/e_attributes_llava.jsonl" \
    --image_root "/mnt/tidalfs-hssh01/dataset/mmeb/COCO" \
    --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
    --model_id "checkpoints/LamRA-Ret"

BOX_OP='crop-lamra' LAYERWISE=0 CUDA_VISIBLE_DEVICES='5' accelerate launch --main_process_port 29504 --multi_gpu eval/eval_fgovd.py \
    --jsonl_path "/mnt/tidalfs-hssh01/usr/liangxun/ICLR26/FG-CLIP/data/fgovd/m_attributes_llava.jsonl" \
    --image_root "/mnt/tidalfs-hssh01/dataset/mmeb/COCO" \
    --original_model_id "/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/" \
    --model_id "checkpoints/LamRA-Ret"