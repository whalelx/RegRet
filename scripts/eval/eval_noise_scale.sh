IMAGE_PATH_PREFIX=$(pwd | awk -F'/usr/' '{print $1}')/dataset/mmeb/M-BEIR
# MODEL_ID="./checkpoints/LamRA-Ret"
# MODEL_ID="./checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs"
# MODEL_ID="./checkpoints/qwen2-vl-7b_abla_vpt_585ktxt_tune_stage2-largebs"
# MODEL_ID="./checkpoints/qwen2-vl-7b_abla_lamra_tune_fgmb16gpu-largebs"

# MODEL_ID="checkpoints/Lemur_7B_Pretrain_585ktxt_final"
MODEL_ID="checkpoints/qwen2-vl-7b_Lemur_585ktxt_tune_stage2-largebs"
# MODEL_ID="../LamRA/checkpoints/LamRA-Ret/"

ORIGINAL_MODEL_ID=/mnt/tidalfs-hssh01/usr/liangxun/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac/ 

if [ -n "$1" ]; then
    MODEL_ID="$1"
fi

export LAYERWISE=1
export BOXOP="crop"

export BOX_NOISE_TYPE=scale
export BOX_NOISE_STD_RATIO=0.08

# 定义要测试的噪声强度
NOISE_INTENSITIES=( 0.30 )
# 0.02 0.05 0.08 0.12 0.15
# image	image	deepfashion	16	Retrieve the same object as the one bounded by the red box.	Determine the content that matches the bounded object in the image.	I want to find the image that corresponds to the region in the red box of the image.	You have to identify the commodity image that matches the object in the red box.

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_xhs.py \
#     --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_deepfashion_task4_test.jsonl \
#     --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_deepfashion_task4_cand_pool.jsonl \
#     --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_deepfashion_task4_cand_pool.jsonl \
#     --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
#     --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_deepfashion_task4_test_qrels.txt \
#     --original_model_id ${ORIGINAL_MODEL_ID} \
#     --image_path_prefix /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/DeepFashion2/validation/image/ \
#     --model_id ${MODEL_ID}

for INTENSITY in "${NOISE_INTENSITIES[@]}"; do
    export BOX_NOISE_INTENSITY=$INTENSITY
    echo "Running with BOX_NOISE_INTENSITY=$BOX_NOISE_INTENSITY"
    
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29519 eval/eval_xhs.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsnote_task4_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsnote_task4_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsnote_task4_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --save_dir ./noise-experiments \
        --save_name ${BOX_NOISE_TYPE}_${BOX_NOISE_INTENSITY}_${BOX_NOISE_STD_RATIO} \
        --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/note_data/20250304/images \
        --model_id ${MODEL_ID}

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task3_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task3_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task3_test_qrels.txt \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --save_dir ./noise-experiments \
        --save_name ${BOX_NOISE_TYPE}_${BOX_NOISE_INTENSITY}_${BOX_NOISE_STD_RATIO} \
        --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images_clean \
        --model_id ${MODEL_ID}

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_dam_task0_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_dam_task0_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_dam_task0_test_qrels.txt \
        --save_dir ./noise-experiments \
        --save_name ${BOX_NOISE_TYPE}_${BOX_NOISE_INTENSITY}_${BOX_NOISE_STD_RATIO} \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/dam_images \
        --model_id ${MODEL_ID}

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fgclip_task3_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task3_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fgclip_task3_test_qrels.txt \
        --save_dir ./noise-experiments \
        --save_name ${BOX_NOISE_TYPE}_${BOX_NOISE_INTENSITY}_${BOX_NOISE_STD_RATIO} \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/fgclip_images \
        --model_id ${MODEL_ID}

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_fgclip_task0_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task0_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_fgclip_task0_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_fgclip_task0_test_qrels.txt \
        --save_dir ./noise-experiments \
        --save_name ${BOX_NOISE_TYPE}_${BOX_NOISE_INTENSITY}_${BOX_NOISE_STD_RATIO} \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/fgclip_images \
        --model_id ${MODEL_ID}

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsgood_task4_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task4_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task4_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsgood_task4_test_qrels.txt \
        --save_dir ./noise-experiments \
        --save_name ${BOX_NOISE_TYPE}_${BOX_NOISE_INTENSITY}_${BOX_NOISE_STD_RATIO} \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/goods_data/from_20250401_to_20250407/images \
        --model_id ${MODEL_ID}


    export BOXOP="concat"
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_imgdiff_task6_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_imgdiff_task6_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_imgdiff_task6_test_qrels.txt \
        --save_dir ./noise-experiments \
        --save_name ${BOX_NOISE_TYPE}_${BOX_NOISE_INTENSITY}_${BOX_NOISE_STD_RATIO} \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX}/../Img-Diff/ \
        --model_id ${MODEL_ID}


    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_xhsgood_task7_merge_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task7_merge_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_xhsgood_task7_merge_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_xhsgood_task7_merge_test_qrels.txt \
        --save_dir ./noise-experiments \
        --save_name ${BOX_NOISE_TYPE}_${BOX_NOISE_INTENSITY}_${BOX_NOISE_STD_RATIO} \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX}/../xhs_data/goods_data/from_20250401_to_20250407/images \
        --model_id ${MODEL_ID}

    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_xhs.py \
        --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_vismin_task6_test.jsonl \
        --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
        --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_vismin_task6_cand_pool.jsonl \
        --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
        --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_vismin_task6_test_qrels.txt \
        --save_dir ./noise-experiments \
        --save_name ${BOX_NOISE_TYPE}_${BOX_NOISE_INTENSITY}_${BOX_NOISE_STD_RATIO} \
        --original_model_id ${ORIGINAL_MODEL_ID} \
        --image_path_prefix ${IMAGE_PATH_PREFIX}/mbeir_images/vismin_images \
        --model_id ${MODEL_ID}
done
