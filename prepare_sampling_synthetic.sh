#!/bin/bash

set -e

echo "=== Creating folders + configs ==="
BASE_DIR="inference_runs/synthetic_math/qwen2.5-0.5b"
STEPS=(200 400 800 1600 3200)
EP=1
RUN_NAMES=()
for STEP in "${STEPS[@]}"; do
    echo $STEP
    # Forward direction
    ckpt_dir=runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1/checkpoint-${STEP}
    SAMPLE_DIR="$BASE_DIR/sft_forward_ep$((STEP / 200))"
    mkdir -p $SAMPLE_DIR
    cat > "$SAMPLE_DIR/sampler_config.yaml" <<EOF
sampler:
  class: VLLMSampler
  model_name: ${ckpt_dir}
  temperature: 1.0
  top_p: 0.95
  top_k: -1
  max_tokens: 512
  trust_remote_code: true
EOF
    RUN_NAMES+=("sft_forward_ep$((STEP / 200))")

    # Reverse direction
    ckpt_dir=runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_reverse_lr1e-5_bs32_ga1/checkpoint-${STEP}
    SAMPLE_DIR="$BASE_DIR/sft_reverse_ep$((STEP / 200))"
    mkdir -p $SAMPLE_DIR
    cat > "$SAMPLE_DIR/sampler_config.yaml" <<EOF
sampler:
  class: VLLMSampler
  model_name: ${ckpt_dir}
  temperature: 1.0
  top_p: 0.95
  top_k: -1
  max_tokens: 512
  trust_remote_code: true
EOF
    RUN_NAMES+=("sft_reverse_ep$((STEP / 200))")
    EP=$((EP + 1))
done

echo "=== Running prompt generation ==="
MODEL_NAME=runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1/checkpoint-200
dir="$BASE_DIR"
dataset="arithchain_2_10"
echo "Running: folder=$folder dataset=$dataset"
python src/inference/build_prompts.py \
  --dataset_name "$dataset" \
  --save_dir "$dir" \
  --tokenizer_path $MODEL_NAME \
  --chat_template_path src/alpaca_template.jira


echo "=== Generating Job list ==="
NUM_SAMPLES=64
JOB_FILE="job_list.txt"
> "$JOB_FILE"
for run in "${RUN_NAMES[@]}"; do
  echo "${BASE_DIR}|${run}|${dataset}|${NUM_SAMPLES}" >> "$JOB_FILE"
done
