IMAGE_PATH_PREFIX=$(pwd | awk -F'/usr/' '{print $1}')/dataset/mmeb/M-BEIR

MODEL_ID="intfloat/mmE5-mllama-11b-instruct" \
CODEBASE=mmE5

# MODEL_ID='nvidia/MM-Embed'
# CODEBASE=mmembed

# MODEL_ID='TIGER-Lab/VLM2Vec-LLaVa-Next'
# CODEBASE=vlm2vec

# python -m ipdb eval/eval_mme5.py \
# CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mme5.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsgood_task4_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task4_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task4_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsgood_task4_test_qrels.txt \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/goods_data/from_20250401_to_20250407/images \
#     --query_modal "image,text" \
#     --cand_modal "image" \
#     --model_name ${MODEL_ID}
# exit
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsgood_task7_merge_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task7_merge_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task7_merge_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsgood_task7_merge_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/goods_data/from_20250401_to_20250407/images \
#     --model_id ${MODEL_ID}

# CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_mme5.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsnote_task4_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsnote_task4_test_qrels.txt \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/note_data/20250304/images \
#     --query_modal "image" \
#     --cand_modal "image" \
#     --model_name ${MODEL_ID}

# CODEBASE=${CODEBASE} python -m ipdb eval/eval_mme5.py \
# CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_mme5.py \
    # --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_vismin_task6_test.jsonl \
    # --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
    # --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
    # --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
    # --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_vismin_task6_test_qrels.txt \
    # --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/vismin_images \
    # --query_modal "image,text" \
    # --cand_modal "image" \
    # --batch_size 40 \
    # --model_name ${MODEL_ID}


# CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_mme5.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task3_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task3_test_qrels.txt \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images \
#     --query_modal "image" \
#     --cand_modal "text" \
#     --model_name ${MODEL_ID}

# CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_mme5.py \
CODEBASE=${CODEBASE} python -m ipdb eval/eval_mme5.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task0_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task0_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images \
    --query_modal "text" \
    --cand_modal "image" \
    --model_name ${MODEL_ID}

# CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_mme5.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fgclip_task3_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fgclip_task3_test_qrels.txt \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/fgclip_images \
#     --query_modal "image" \
#     --cand_modal "text" \
#     --model_name ${MODEL_ID}

# CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_mme5.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fgclip_task0_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task0_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task0_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fgclip_task0_test_qrels.txt \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/fgclip_images \
#     --query_modal "text" \
#     --cand_modal "image" \
#     --model_name ${MODEL_ID}

# CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_mme5.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_imgdiff_task6_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_imgdiff_task6_test_qrels.txt \
#     --image_path_prefix ${IMAGE_PATH_PREFIX}/../Img-Diff/ \
#     --query_modal "image,text" \
#     --cand_modal "image" \
#     --model_name ${MODEL_ID}
