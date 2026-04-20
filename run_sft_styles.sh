#!/usr/bin/env bash

DATASET_NAME=$1
model_short_name=$2
shift 2

DATA_ROOT="/scratch1/hnn5071/workspace/rm-limeval/datasets"

########################
# MODEL CONFIG
########################
declare -A MODEL_CONFIGS

MODEL_CONFIGS["qwen2.5_0.5b"]="unsloth/Qwen2.5-0.5B 32 1 5e-5"
MODEL_CONFIGS["qwen2.5_1.5b"]="unsloth/Qwen2.5-1.5B 32 1 3e-5"
MODEL_CONFIGS["llama3.2_1b"]="unsloth/Llama-3.2-1B 32 1 3e-5"
MODEL_CONFIGS["llama3.2_3b"]="unsloth/Llama-3.2-3B 32 1 3e-5"
MODEL_CONFIGS["gemma3_1b"]="unsloth/gemma-3-1b-pt 32 2 3e-5"
MODEL_CONFIGS["gemma3_4b"]="unsloth/gemma-3-4b-pt 16 2 3e-5"
MODEL_CONFIGS["llama3.1_8b"]="unsloth/Llama-3.1-8B 16 1 3e-5"
MODEL_CONFIGS["evolm-4b"]="zhenting/evolm-4B-160BT-cpt-MixedFW8FM42 8 4 3e-5"

# Validate model
if [ -z "${model_short_name}" ]; then
    echo "Error: model_short_name is not set"
    exit 1
fi

CONFIG="${MODEL_CONFIGS[$model_short_name]}"

if [ -z "$CONFIG" ]; then
    echo "Invalid model name. Options are:"
    printf "  - %s\n" "${!MODEL_CONFIGS[@]}"
    exit 1
fi

# Parse model config
read -r MODEL_NAME BATCH_SIZE GRAD_ACCUM LR <<< "$CONFIG"

TOTAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM))

########################
# DATASET CONFIG
########################
declare -A DATA_PATHS
declare -A DATA_SIZES   # for dynamic SAVE_STEPS
DATA_PATHS["gsm8k_datasetlevelcombined"]="${DATA_ROOT}/mixing_styles/final/Mar20/gsm8k_datasetlevelcombined.parquet"
DATA_PATHS["gsm8k_single_mix"]="${DATA_ROOT}/mixing_styles/final_v2/gsm8k_train_single_mix.parquet"

# Dataset sizes (only needed when computing dynamically)
DATA_SIZES["mathgsm8k_nlreasoning"]=23694
DATA_SIZES["mathgsm8k_code"]=23694
DATA_SIZES["gsm8k_nlreasoning"]=12850
DATA_SIZES["gsm8k_code"]=12850
DATA_SIZES["gsm8k_datasetlevelcombined"]=12850
DATA_SIZES["gsm8k_single_mix"]=6400

# Validate dataset
if [ -z "${DATASET_NAME}" ]; then
    echo "Error: DATASET_NAME is not set"
    exit 1
fi

DATA_PATH="${DATA_PATHS[$DATASET_NAME]}"
DATA_SIZE="${DATA_SIZES[$DATASET_NAME]}"

if [ -z "$DATA_PATH" ]; then
    echo "Invalid dataset name. Options are:"
    printf "  - %s\n" "${!DATA_PATHS[@]}"
    exit 1
fi

########################
# SAVE STEPS LOGIC
########################

# Default: compute from dataset size
SAVE_STEPS=$((DATA_SIZE / TOTAL_BATCH_SIZE))

# Override for fixed cases (to match your original logic exactly)
case "${DATASET_NAME}" in
  "mathgsm8k_nlreasoning" | "mathgsm8k_code")
    SAVE_STEPS=186
    ;;
  "gsm8k_nlreasoning" | "gsm8k_code" | "gsm8k_datasetlevelcombined")
    SAVE_STEPS=101
    ;;
  "gsm8k_single_mix")
    # already computed dynamically (6400 / TOTAL_BATCH_SIZE)
    ;;
esac

########################
# DEBUG PRINT
########################
echo "MODEL_NAME: $MODEL_NAME"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "GRAD_ACCUM: $GRAD_ACCUM"
echo "LR: $LR"
echo "TOTAL_BATCH_SIZE: $TOTAL_BATCH_SIZE"

echo "DATASET_NAME: $DATASET_NAME"
echo "DATA_PATH: $DATA_PATH"
echo "DATA_SIZE: $DATA_SIZE"
echo "SAVE_STEPS: $SAVE_STEPS"


NUM_TRAIN_EPOCHS=8
WANDB_PROJECT="reasoning_style_sft"
RUN_NAME="unsloth_${model_short_name}_sft_ep8_${DATASET_NAME}_lr${LR}_bs${BATCH_SIZE}_ga${GRAD_ACCUM}"

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
  --response_key "solution" \
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