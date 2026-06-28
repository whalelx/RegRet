NUM_GPUS=8
NNODES=1
NODE_RANK=0 # 1
MASTER_ADDR="0.0.0.0"
MASTER_PORT=29508

DISTRIBUTED_ARGS="
    --nnodes=${NNODES} \
    --nproc_per_node ${NUM_GPUS} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

# DISTRIBUTED_ARGS="
#     --nnodes=${NNODES} \
#     --nproc_per_node ${NUM_GPUS} \
#     --master_port ${MASTER_PORT}
# "

# arguments that are very likely to be changed
# according to your own case
MODEL_ID=${3:-"qwen2_5-vl-7b"}
DATASET_PATH=$(pwd | awk -F'/usr/' '{print $1}')/dataset/mmeb
IMAGE_PATH_PREFIX=${DATASET_PATH}/M-BEIR
QUERY_DATA_PATH=${IMAGE_PATH_PREFIX}/query/union_train/mbeir_union_up_train.jsonl
CAND_POOL_PATH=${IMAGE_PATH_PREFIX}/cand_pool/global/mbeir_union_train_cand_pool.jsonl
INSTRUCTIONS_PATH=${IMAGE_PATH_PREFIX}/instructions/query_instructions.tsv
MODEL_LOCAL_PATH=${2:-"./checkpoints/qwen2_5-vl-7b_dam_cvp_fixwin_tune"}

TRAIN_VISION_ENCODER=True                              
USE_VISION_LORA=False                              
TRAIN_VISION_PROJECTOR=True          

USE_LORA=True                                           
Q_LORA=False                                           
LORA_R=128                                                
LORA_ALPHA=256                                           
RUN_ID=${1:-${MODEL_ID}_fixwin_cp}

DS_STAGE=zero3
PER_DEVICE_BATCH_SIZE=6
GRAD_ACCUM=1                                   
NUM_EPOCHS=1       

LR=1e-4
MODEL_MAX_LEN=1024


torchrun $DISTRIBUTED_ARGS train/pretrain_vision.py \
    --model_id $MODEL_ID \
    --query_data_path $QUERY_DATA_PATH \
    --cand_pool_path $CAND_POOL_PATH \
    --instructions_path $INSTRUCTIONS_PATH \
    --xhs_query_data_path ${DATASET_PATH}/M-BEIR/query/train/mbeir_xhsnote_task7_train.jsonl \
    --xhs_cand_pool_path ${DATASET_PATH}/M-BEIR/cand_pool/local/mbeir_xhsnote_task7_cand_pool.jsonl \
    --dam_data_path ${DATASET_PATH}/describe-anything-data \
    --dam_max_samples 0 \
    --fgclip_data_path ${DATASET_PATH}/fg-clip \
    --fgclip_max_samples 245500 \
    --output_dir ./checkpoints/$RUN_ID \
    --report_to tensorboard \
    --run_name $RUN_ID \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 20 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --model_local_path $MODEL_LOCAL_PATH \
    --image_path_prefix $IMAGE_PATH_PREFIX \
    --use_flash_attn True
