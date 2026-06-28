IMAGE_PATH_PREFIX=$(pwd | awk -F'/usr/' '{print $1}')/dataset/mmeb/M-BEIR
MODEL_ID="../test_cp_tune/tmp_ckpts/hhh"
MODEL_ID="../test_cp_tune/tmp_ckpts/damret"
# MODEL_ID="./checkpoints/qwen2_5-vl-7b_damret_vismin"

ORIGINAL_MODEL_ID=./checkpoints/LEMUIR_Pretrain

# PROMPT_TSV=query_instructions
# PROMPT_TSV=query_instructions_box
PROMPT_TSV=query_instructions_concat

# MODEL_ID="./tmp_ckpts/dam_pretrain+lamrallm"
# MODEL_ID="checkpoints/qwen2_5-vl-7b_LEMUIR_tune_genloss0.3_mbeirlanguage"

if [ -n "$1" ]; then
    MODEL_ID="$1"
fi

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_vismin_task6_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/${PROMPT_TSV}.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_vismin_task6_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/vismin_images \
#     --model_id ${MODEL_ID}

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_imgdiff_task6_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/${PROMPT_TSV}.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_imgdiff_task6_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/../Img-Diff/ \
#     --model_id ${MODEL_ID}

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsnote_task4_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/${PROMPT_TSV}.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsnote_task4_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/note_data/20250304/images \
#     --model_id ${MODEL_ID}




# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task3_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/${PROMPT_TSV}.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task3_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images \
#     --model_id ${MODEL_ID}

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task0_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/${PROMPT_TSV}.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task0_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images \
#     --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fgclip_task3_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/${PROMPT_TSV}.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fgclip_task3_test_qrels.txt \
    --original_model_id ${ORIGINAL_MODEL_ID} \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/fgclip_images \
    --model_id ${MODEL_ID}