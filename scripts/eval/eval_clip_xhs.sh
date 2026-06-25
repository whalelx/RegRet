IMAGE_PATH_PREFIX=$(pwd | awk -F'/usr/' '{print $1}')/dataset/mmeb/M-BEIR
# clip
# MODEL_ID="openai/clip-vit-large-patch14"
# siglip
MODEL_ID="google/siglip-so400m-patch14-384"
# siglip2
# MODEL_ID="google/siglip2-so400m-patch16-naflex"
# eva-clip
# MODEL_ID="https://hf-mirror.com/BAAI/EVA-CLIP-8B"
# MODEL_ID="BAAI/EVA-CLIP-8B"

# Blip2
# MODEL_ID="Salesforce/blip2-itm-vit-g"
# MODEL_ID="Salesforce/blip-itm-large-coco"
# https://github.com/salesforce/LAVIS?tab=readme-ov-file#model-zoo
# MODEL_ID="https://hf-mirror.com/Salesforce/blip-itm-large-coco"

# mme5和vlm2vec是一脉相承的，搞定了一个另一个也ok了: 但是要研究一下

# fgclip： 
# 可以像clip一样直接import，https://github.com/360CVGroup/FG-CLIP/blob/main/fgclip/eval/in_1K/coco_box_cls.py
MODEL_ID="qihoo360/fg-clip-base"
IMAGESIZE="--image_size 224"

export BOXOP="draw"

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_clip.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsgood_task4_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task4_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task4_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsgood_task4_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/goods_data/from_20250401_to_20250407/images \
    --query_modal "image" \
    --cand_modal "image" \
    ${IMAGESIZE} \
    --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_clip.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsgood_task7_merge_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task7_merge_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task7_merge_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsgood_task7_merge_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/goods_data/from_20250401_to_20250407/images \
    --query_modal "image,text" \
    --cand_modal "image" \
    ${IMAGESIZE} \
    --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_clip.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsnote_task4_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsnote_task4_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/note_data/20250304/images \
    --query_modal "image" \
    --cand_modal "image" \
    ${IMAGESIZE} \
    --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_clip.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_vismin_task6_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_vismin_task6_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/vismin_images \
    --query_modal "image,text" \
    --cand_modal "image" \
    ${IMAGESIZE} \
    --model_id ${MODEL_ID}


CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_clip.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task3_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task3_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images_clean \
    --query_modal "image" \
    --cand_modal "text" \
    ${IMAGESIZE} \
    --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_clip.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task0_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task0_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images \
    --query_modal "text" \
    --cand_modal "image" \
    ${IMAGESIZE} \
    --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_clip.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fgclip_task3_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fgclip_task3_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/fgclip_images \
    --query_modal "image" \
    --cand_modal "text" \
    ${IMAGESIZE} \
    --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_clip.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fgclip_task0_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task0_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task0_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fgclip_task0_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/fgclip_images \
    --query_modal "text" \
    --cand_modal "image" \
    ${IMAGESIZE} \
    --model_id ${MODEL_ID}

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29530 eval/eval_clip.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_imgdiff_task6_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions_box.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_imgdiff_task6_test_qrels.txt \
    --image_path_prefix ${IMAGE_PATH_PREFIX}/../Img-Diff/ \
    --query_modal "image,text" \
    --cand_modal "image" \
    ${IMAGESIZE} \
    --model_id ${MODEL_ID}
