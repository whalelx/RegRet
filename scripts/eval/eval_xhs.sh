IMAGE_PATH_PREFIX=/mnt/tidalfs-hssh01/dataset/mmeb/M-BEIR
# MODEL_ID="./checkpoints/qwen2_5-vl-7b_LEMUIR_tune_genloss0.3"
# MODEL_ID="checkpoints/qwen2_5-vl-7b_LEMUIR_tune"
# MODEL_ID="checkpoints/qwen2_5-vl-7b_LEMUIR_tune_nodam"  # basline
MODEL_ID="./tmp_ckpts/dam_cvp"  # language

ORIGINAL_MODEL_ID=./checkpoints/LEMUIR_Pretrain

# MODEL_ID="./tmp_ckpts/dam_pretrain+lamrallm"
# MODEL_ID="checkpoints/qwen2_5-vl-7b_LEMUIR_tune_genloss0.3_mbeirlanguage"


CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhs_task7_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_xhs_task7_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_xhs_task7_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhs_task7_test_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --image_path_prefix ${IMAGE_PATH_PREFIX} \
    --model_id ${MODEL_ID}

# MODEL_ID="/mnt/tidal-alsh01/dataset/mmeb/LamRA-Ret-Qwen2.5VL-7b" #"./checkpoints/LamRA-Ret"
# ORIGINAL_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
# ORIGINAL_MODEL_ID=/mnt/tidal-alsh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/
