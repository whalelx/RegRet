# MODEL_ID="./checkpoints/qwen2_5-vl-7b_LEMUIR_tune_genloss0.3"
# todo 修改qwen2。5的路径
# MODEL_ID=/mnt/tidal-alsh01/dataset/mmeb/LamRA-Ret-Qwen2.5VL-7b
# MODEL_ID="../lemuir/checkpoints/qwen2_5-vl-7b_LEMUIR_tune_nodam"
# MODEL_ID="./tmp_ckpts/dam_globaldt"  # language
# MODEL_ID="./checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-h20"
# MODEL_ID="./checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs"
MODEL_ID="checkpoints/qwen2-vl-7b_Lemur_585ktxt_final_tune_stage2-largebs"
MODEL_ID="./checkpoints/qwen2-vl-7b_abla_vpt_585ktxt_tune_stage2-largebs"

ORIGINAL_MODEL_ID=/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/ 

# MODEL_ID="checkpoints/LEMUIR_xPretrain"
# MODEL_ID="checkpoints/LEMUIR_Pretrain"
# MODEL_ID="checkpoints/qwen2_5-vl-7b_LEMUIR_tune_genloss0.3_mbeirlanguage"

IMAGE_PATH_PREFIX=/mnt/tidalfs-hssh01/dataset/mmeb/M-BEIR

if [ -n "$1" ]; then
    MODEL_ID="$1"
    shift 1
fi

# MODEL_ID="/mnt/tidal-alsh01/dataset/mmeb/LamRA-Ret-Qwen2.5VL-7b" #"./checkpoints/LamRA-Ret"
# ORIGINAL_MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
# IMAGE_PATH_PREFIX=/mnt/tidal-alsh01/dataset/mmeb/M-BEIR

# MODEL_ID="checkpoints/qwen2_5-vl-7b_MetaQuery_Pretrain-Qwen2.5VL" #"./checkpoints/LamRA-Ret"
# ORIGINAL_MODEL_ID=./checkpoints/LEMUIR_Pretrain

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir_fasis.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_mscoco_task0_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_mscoco_task0_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID} $@


# bug
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_mbeir_fasis.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_oven_task6_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_oven_task6_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_oven_task6_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID} $@

########### used in paper

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir_fasis.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_infoseek_task6_test.jsonl \
#     --query_cand_pool ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_infoseek_task6_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID} $@


# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir_fasis.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_oven_task8_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_oven_task8_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_oven_task8_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID} $@

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir_fasis.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_infoseek_task8_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_infoseek_task8_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID} $@
