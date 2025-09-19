MODEL_ID="./checkpoints/qwen2-vl-7b_Lemur_585ktxt_nocausal_tune_stage2-largebs"

ORIGINAL_MODEL_ID=/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/ 

IMAGE_PATH_PREFIX=/mnt/tidalfs-hssh01/dataset/mmeb/M-BEIR




#!/bin/bash

# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <group_number>"
    echo "Group numbers: 1, 2, or 3"
    exit 1
fi

group=$1

execute_group1() {
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fashioniq_task7_test.jsonl \
        --query_cand_pool ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fashioniq_task7_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fashion200k_task3_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_fashion200k_task3_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fashion200k_task3_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fashion200k_task0_test.jsonl \
        --query_cand_pool ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fashion200k_task0_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_edis_task2_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_edis_task2_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_edis_task2_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@
}
##############
execute_group2() {
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_cirr_task7_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_cirr_task7_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_cirr_task7_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_nights_task4_test.jsonl \
        --query_cand_pool ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_nights_task4_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_nights_task4_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_visualnews_task0_test.jsonl \
        --query_cand_pool ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_visualnews_task0_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_webqa_task2_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_webqa_task2_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_webqa_task2_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@
}
##############

execute_group3() {

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_mscoco_task0_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_mscoco_task0_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_mscoco_task3_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_mscoco_task3_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_visualnews_task3_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_visualnews_task3_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_visualnews_task3_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_webqa_task1_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_webqa_task1_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_webqa_task1_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX} \
        --model_id ${MODEL_ID} $@
}

case $group in
    1)
        echo "Executing Group 1"
        execute_group1
        ;;
    2)
        echo "Executing Group 2"
        execute_group2
        ;;
    3)
        echo "Executing Group 3"
        execute_group3
        ;;
    *)
        echo "Invalid group number. Please use 1, 2, or 3."
        exit 1
        ;;
esac




# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29508 eval/eval_mbeir.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_oven_task6_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_oven_task6_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_oven_task6_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID} $@

# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_infoseek_task6_test.jsonl \
#     --query_cand_pool ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_infoseek_task6_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID} $@


# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_oven_task8_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_oven_task8_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_oven_task8_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID} $@

# CUDA_VISIBLE_DEVICES='0' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mbeir.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_infoseek_task8_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_infoseek_task8_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX} \
#     --model_id ${MODEL_ID} $@
