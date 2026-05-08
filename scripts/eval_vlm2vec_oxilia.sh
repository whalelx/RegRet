IMAGE_PATH_PREFIX=$(pwd | awk -F'/usr/' '{print $1}')/dataset/mmeb/M-BEIR

MODEL_ID='TIGER-Lab/VLM2Vec-LLaVa-Next'
CODEBASE=vlm2vec


export BOXOP="crop"

# CODEBASE=${CODEBASE}  CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mme5_oxilia.py \
#   --eval_task ilias \
#   --model_name ${MODEL_ID} \
#   --ilias_dataset_path /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/ilias \
#   --pooling last --normalize True \
#   --ilias_eval_mode image  # or "text"


# CODEBASE=${CODEBASE}  CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mme5_oxilia.py \
#   --eval_task ilias \
#   --model_name ${MODEL_ID} \
#   --ilias_dataset_path /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/ilias \
#   --pooling last --normalize True \
#   --batch_size 48 \
#   --ilias_eval_mode text


# CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_mme5_oxilia.py \
#   --eval_task oxford \
#   --model_name ${MODEL_ID} \
#   --oxford_dataset_script dataset/dataset_oxford5k.py \
#   --batch_size 48 \
#   --pooling last --normalize True \
#   --oxford_dataset_name roxford5k  # or "rparis6k"


## --- 以上代码可以用mme5跑，但是下面的不行！
CODEBASE=${CODEBASE} CUDA_VISIBLE_DEVICES='7' accelerate launch --multi_gpu --main_process_port 29509 eval/eval_vlm2vec.py \
    --query_data_path ${IMAGE_PATH_PREFIX}/query/test/mbeir_deepfashion_task4_test.jsonl \
    --query_cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_deepfashion_task4_cand_pool.jsonl \
    --cand_pool_path ${IMAGE_PATH_PREFIX}/cand_pool/test/mbeir_deepfashion_task4_cand_pool.jsonl \
    --instructions_path ${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv \
    --qrels_path ${IMAGE_PATH_PREFIX}/qrels/test/mbeir_deepfashion_task4_test_qrels.txt \
    --image_path_prefix /mnt/tidalfs-hssh01/dataset/mmeb/RegionLevel/DeepFashion2/validation/image/ \
    --query_modal "image" \
    --cand_modal "image" \
    --batch_size 30 \
    --model_name ${MODEL_ID} --pooling last --normalize True 