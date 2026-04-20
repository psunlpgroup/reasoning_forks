#!/bin/bash

set -x

DATASET_NAME=$1
model_short_name=$2

shift 2

# --------------------------------------------------
# Configuration (edit these)
# --------------------------------------------------
DATA_ROOT="./datasets/math_graph"

DATASET_NAMES=(
    "pathstar_2_10_forward"
    "pathstar_2_10_reverse"
)
# If DATASET_NAME is not in DATASET_NAMES, exit
if [[ ! "${DATASET_NAMES[@]}" =~ "${DATASET_NAME}" ]]; then
    echo "Invalid dataset name. Please choose from: ${DATASET_NAMES[@]}"
    exit 1
fi

# TOTAL BATCH SIZE = BATCH_SIZE * GRAD_ACCUM = 128
if [ "${DATASET_NAME}" == "pathstar_2_10_forward" ]; then
    DATA_PATH="${DATA_ROOT}/pathstar_2_10/train_forward.parquet" # 6400 samples
    SAVE_STEPS=100
elif [ "${DATASET_NAME}" == "pathstar_2_10_reverse" ]; then
    DATA_PATH="${DATA_ROOT}/pathstar_2_10/train_reverse.parquet" # 6400 samples
    SAVE_STEPS=100
fi


LR=1e-5
BATCH_SIZE=64
GRAD_ACCUM=1

MODEL_NAMES=(
    "qwen2.5_0.5b"
    "evolm_1b"
)
if [ "${model_short_name}" == "qwen2.5_0.5b" ]; then
    MODEL_NAME="unsloth/Qwen2.5-0.5B"
elif [ "${model_short_name}" == "evolm_1b" ]; then
    MODEL_NAME="zhenting/evolm-1B-160BT-cpt-MixedFW8FM42"
else
    echo "Invalid model name. Please choose from: ${MODEL_NAMES[@]}"
    exit 1
fi


NUM_TRAIN_EPOCHS=16

WANDB_PROJECT="reasoning_sft_graph"
RUN_NAME="unsloth_${model_short_name}_sft_ep16_${DATASET_NAME}_lr${LR}_bs${BATCH_SIZE}_ga${GRAD_ACCUM}"

OUTPUT_DIR="runs/${WANDB_PROJECT}/${RUN_NAME}"

# --------------------------------------------------
# Environment
# --------------------------------------------------

export TOKENIZERS_PARALLELISM=false

mkdir -p ${OUTPUT_DIR}

LOG_FILE="${OUTPUT_DIR}/train.log"

echo "Starting training..."
echo "Run name: ${RUN_NAME}"
echo "Logs: ${LOG_FILE}"

# --------------------------------------------------
# Launch
# --------------------------------------------------

python src/sft.py \
  --model_name ${MODEL_NAME} \
  --data_path ${DATA_PATH} \
  --prompt_key "question" \
  --response_key "answer" \
  --batch_size ${BATCH_SIZE} \
  --grad_accum ${GRAD_ACCUM} \
  --warmup_ratio 0.1 \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --learning_rate ${LR} \
  --save_steps ${SAVE_STEPS} \
  --output_dir ${OUTPUT_DIR} \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --wandb_run_name ${RUN_NAME} $@ \
  2>&1 | tee ${LOG_FILE}

echo "Training finished."