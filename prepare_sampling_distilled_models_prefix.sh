#!/bin/bash

set -e

BASE_DIR="inference_runs/ds-distill/ds-qwen-1.5b"
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Datasets to run
DATASETS=(
  "math500"
  "aime24"
  "aime25"
  "gsm8k"
  "arithchain_2_10"
  "counterfact_addition"
  "capitalQA"
)

# Folder → prefix mapping
declare -A folders=(
  ["prefix_Okay"]="Okay"
  ["prefix_Let"]="Let"
  ["prefix_Alright"]="Alright"
  ["default"]=""
  ["prefix_To"]="To"
)

echo "=== Creating folders + configs ==="

for folder in "${!folders[@]}"; do
  dir="$BASE_DIR/$folder"
  mkdir -p "$dir"

  prefix="${folders[$folder]}"
  config_file="$dir/sampler_config.yaml"

  if [ -z "$prefix" ]; then
    cat > "$config_file" <<EOF
sampler:
  class: VLLMSampler
  model_name: $MODEL_NAME
  temperature: 0.6
  top_p: 0.95
  top_k: -1
  max_tokens: 32768
  max_model_len: 32768
  trust_remote_code: true
EOF
  else
    cat > "$config_file" <<EOF
sampler:
  class: VLLMSampler
  model_name: $MODEL_NAME
  temperature: 0.6
  top_p: 0.95
  top_k: -1
  max_tokens: 32768
  max_model_len: 32768
  trust_remote_code: true
  thinking_prefix: $prefix
EOF
  fi
done

echo "=== Running prompt generation ==="


dir="$BASE_DIR"

for dataset in "${DATASETS[@]}"; do
  echo "Running: folder=$folder dataset=$dataset"

  python src/inference/build_prompts.py \
    --dataset_name "$dataset" \
    --save_dir "$dir" \
    --apply_chat_template \
    --tokenizer_path $MODEL_NAME
done


echo "=== Generating Job list ==="

RUN_NAMES=("default" "prefix_Alright" "prefix_Let" "prefix_Okay" "prefix_To")
NUM_SAMPLES=8
JOB_FILE="job_list.txt"
> "$JOB_FILE"

for split in "${DATASETS[@]}"; do
  for run in "${RUN_NAMES[@]}"; do
    echo "${BASE_DIR}|${run}|${split}|${NUM_SAMPLES}" >> "$JOB_FILE"
  done
done
