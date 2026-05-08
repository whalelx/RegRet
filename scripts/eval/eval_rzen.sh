#!/bin/bash
# Evaluate RzenEmbed on MMEB / iLIAS / Oxford-Paris
# Usage:
#   bash scripts/eval/eval_rzen.sh [model_path]
#
# Examples:
#   # MMEB evaluation
#   EVAL_TASK=mmeb bash scripts/eval/eval_rzen.sh qihoo360/RzenEmbed
#
#   # Oxford evaluation only
#   EVAL_TASK=oxford bash scripts/eval/eval_rzen.sh qihoo360/RzenEmbed
#
#   # iLIAS evaluation only
#   EVAL_TASK=ilias bash scripts/eval/eval_rzen.sh qihoo360/RzenEmbed

MODEL_ID="${1:-qihoo360/RzenEmbed}"
EVAL_TASK="${EVAL_TASK:-all}"   # all / mmeb / oxford / ilias
BATCH_SIZE="${BATCH_SIZE:-75}"

IMAGE_PATH_PREFIX=/mnt/tidalfs-hssh01/dataset/mmeb/M-BEIR

# ==================== MMEB ====================
run_mmeb() {
    local QUERY_DATA=$1
    local QUERY_CAND_POOL=$2
    local CAND_POOL=$3
    local INST_PATH=$4
    local QRELS=$5
    local IMG_PREFIX=$6
    shift 6

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 eval/eval_rzen.py \
        --eval_task mmeb \
        --model_name ${MODEL_ID} \
        --batch_size ${BATCH_SIZE} \
        --query_data_path ${QUERY_DATA} \
        --query_cand_pool_path ${QUERY_CAND_POOL} \
        --cand_pool_path ${CAND_POOL} \
        --instructions_path ${INST_PATH} \
        --qrels_path ${QRELS} \
        --image_path_prefix ${IMG_PREFIX} \
        "$@"
}

if [ "$EVAL_TASK" = "mmeb" ] || [ "$EVAL_TASK" = "all" ]; then
    echo "===== Running MMEB evaluations ====="

#    run_mmeb \
#         ${IMAGE_PATH_PREFIX}/query/test/mbeir_deepfashion_task4_test.jsonl \
#         ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_deepfashion_task4_cand_pool.jsonl \
#         ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_deepfashion_task4_cand_pool.jsonl \
#         ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#         ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_deepfashion_task4_test_qrels.txt \
#         ${IMAGE_PATH_PREFIX}/../RegionLevel/DeepFashion2/validation/image/ 

#     run_mmeb \
#         ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsnote_task4_test.jsonl \
#         ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
#         ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
#         ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#         ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsnote_task4_test_qrels.txt \
#         ${IMAGE_PATH_PREFIX}/../xhs_data/note_data/20250304/images


#     run_mmeb \
#         ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsgood_task4_test.jsonl \
#         ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task4_cand_pool.jsonl \
#         ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task4_cand_pool.jsonl \
#         ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#         ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsgood_task4_test_qrels.txt \
#         ${IMAGE_PATH_PREFIX}/../xhs_data/goods_data/from_20250401_to_20250407/images



    # run_mmeb \
    #     ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task3_test.jsonl \
    #     ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
    #     ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
    #     ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    #     ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task3_test_qrels.txt \
    #     ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images_clean


    # run_mmeb \
    #     ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task0_test.jsonl \
    #     ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
    #     ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
    #     ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    #     ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task0_test_qrels.txt \
    #     ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images


    # run_mmeb \
    #     ${IMAGE_PATH_PREFIX}/query/test/mbeir_fgclip_task3_test.jsonl \
    #     ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
    #     ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
    #     ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    #     ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fgclip_task3_test_qrels.txt \
    #     ${IMAGE_PATH_PREFIX}/mbeir_images/fgclip_images


    # run_mmeb \
    #     ${IMAGE_PATH_PREFIX}/query/test/mbeir_fgclip_task0_test.jsonl \
    #     ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task0_cand_pool.jsonl \
    #     ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task0_cand_pool.jsonl \
    #     ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    #     ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fgclip_task0_test_qrels.txt \
    #     ${IMAGE_PATH_PREFIX}/mbeir_images/fgclip_images

    # # export BOXOP="concat"
    run_mmeb \
        ${IMAGE_PATH_PREFIX}/query/test/mbeir_imgdiff_task6_test.jsonl \
        ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
        ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
        ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
        ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_imgdiff_task6_test_qrels.txt \
        ${IMAGE_PATH_PREFIX}/../Img-Diff/



    run_mmeb \
        ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsgood_task7_merge_test.jsonl \
        ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task7_merge_cand_pool.jsonl \
        ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task7_merge_cand_pool.jsonl \
        ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
        ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsgood_task7_merge_test_qrels.txt \
        ${IMAGE_PATH_PREFIX}/../xhs_data/goods_data/from_20250401_to_20250407/images


    run_mmeb \
        ${IMAGE_PATH_PREFIX}/query/test/mbeir_vismin_task6_test.jsonl \
        ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
        ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
        ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
        ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_vismin_task6_test_qrels.txt \
        ${IMAGE_PATH_PREFIX}/mbeir_images/vismin_images
fi

# ==================== Oxford / Paris ====================
if [ "$EVAL_TASK" = "oxford" ] || [ "$EVAL_TASK" = "all" ]; then
    echo "===== Running Oxford/Paris evaluations ====="

    python eval/eval_rzen.py \
        --eval_task oxford \
        --model_name ${MODEL_ID} \
        --batch_size ${BATCH_SIZE} \
        --oxford_dataset_script dataset/dataset_oxford5k.py \
        --oxford_dataset_name roxford5k \
        --oxford_box_op crop
fi

# ==================== iLIAS ====================
if [ "$EVAL_TASK" = "ilias" ] || [ "$EVAL_TASK" = "all" ]; then
    echo "===== Running iLIAS evaluations ====="
    ILIAS_DATASET_PATH="${ILIAS_DATASET_PATH:-/mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/ilias}"

    python eval/eval_rzen.py \
        --eval_task ilias \
        --model_name ${MODEL_ID} \
        --batch_size ${BATCH_SIZE} \
        --ilias_dataset_path ${ILIAS_DATASET_PATH} \
        --ilias_eval_mode image

    python eval/eval_rzen.py \
        --eval_task ilias \
        --model_name ${MODEL_ID} \
        --batch_size ${BATCH_SIZE} \
        --ilias_dataset_path ${ILIAS_DATASET_PATH} \
        --ilias_eval_mode text
fi

echo "===== Done ====="
