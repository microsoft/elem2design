#!/bin/bash

DEEPSPEED=${1}
BASE_MODEL=${2}
DATA_PATH=${3}
IMAGE_FOLDER=${4}
OUTPUT_DIR=${5}
NUM_EPOCH=${6}
BATCH_SIZE=${7}
ACCUMULATION=${8}
SAVE_STEP=${9}
LR=${10}
PRO_LR=${11}
SELECT_FEATURE=${12}
WANDB_NAME=${13}

if [ $DEEPSPEED -gt 0 ]; then
    COMMAND="deepspeed llava/train/train_mem.py --deepspeed ./scripts/zero2.json"
else
    COMMAND="python llava/train/train_mem.py"
fi

$COMMAND \
    --model_name_or_path $BASE_MODEL \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATION \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEP \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --mm_vision_select_feature $SELECT_FEATURE \
    --mm_projector_lr $PRO_LR \
    --run_name $WANDB_NAME