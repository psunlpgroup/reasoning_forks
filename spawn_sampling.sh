#!/bin/bash

trap 'echo "Cleaning up..."; kill -9 -$$ 2>/dev/null' EXIT

# -----------------------------
# Define available GPUs (separate)
# -----------------------------
AVAILABLE_GPUS=(6 7)

# Define jobs: "JOB_DIR|RUN_NAME|SPLIT|NUM_SAMPLES"
JOB_SPLIT_PAIRS=(
    "inference_runs/ds-distill/ds-qwen-1.5b|default|aime24_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Alright|aime24_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Let|aime24_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Okay|aime24_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_To|aime24_test|8"

    "inference_runs/ds-distill/ds-qwen-1.5b|default|aime25_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Alright|aime25_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Let|aime25_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Okay|aime25_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_To|aime25_test|8"

    "inference_runs/ds-distill/ds-qwen-1.5b|default|gsm8k_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Alright|gsm8k_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Let|gsm8k_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Okay|gsm8k_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_To|gsm8k_test|8"

    "inference_runs/ds-distill/ds-qwen-1.5b|default|math500_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Alright|math500_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Let|math500_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Okay|math500_test|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_To|math500_test|8"

    "inference_runs/ds-distill/ds-qwen-1.5b|default|arithmetic_counterfact|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Alright|arithmetic_counterfact|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Let|arithmetic_counterfact|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Okay|arithmetic_counterfact|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_To|arithmetic_counterfact|8"

    "inference_runs/ds-distill/ds-qwen-1.5b|default|capitalQA|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Alright|capitalQA|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Let|capitalQA|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_Okay|capitalQA|8"
    "inference_runs/ds-distill/ds-qwen-1.5b|prefix_To|capitalQA|8"
)

# -----------------------------
# Step 1: Build job queue
# -----------------------------
JOBS=()

for PAIR in "${JOB_SPLIT_PAIRS[@]}"; do
    IFS="|" read -r JOB_DIR RUN_NAME SPLIT NUM_SAMPLES <<< "$PAIR"
    echo "$JOB_DIR/$RUN_NAME"
    mapfile -t SAMPLE_DIRS < <(find "$JOB_DIR" -mindepth 1 -maxdepth 4 -type d \
        -path "$JOB_DIR/$RUN_NAME" | sort)

    for SAMPLE_DIR in "${SAMPLE_DIRS[@]}"; do
        SAMPLER_CONFIG_DIR=$(realpath --relative-to="$JOB_DIR" "$SAMPLE_DIR")

        JOBS+=("$JOB_DIR|$SPLIT|$SAMPLER_CONFIG_DIR|$NUM_SAMPLES")
    done
done

echo "Total jobs: ${#JOBS[@]}"
echo "Available GPUs: ${AVAILABLE_GPUS[*]}"

# -----------------------------
# Step 2: Scheduler
# -----------------------------
declare -A GPU_PIDS

JOB_IDX=0
TOTAL_JOBS=${#JOBS[@]}

while [ "$JOB_IDX" -lt "$TOTAL_JOBS" ]; do
    for GPU in "${AVAILABLE_GPUS[@]}"; do

        # Skip if GPU is busy
        if [[ -n "${GPU_PIDS[$GPU]}" ]]; then
            if kill -0 "${GPU_PIDS[$GPU]}" 2>/dev/null; then
                continue
            fi
        fi

        # Assign next job
        JOB="${JOBS[$JOB_IDX]}"
        IFS="|" read -r JOB_DIR SPLIT SAMPLER_CONFIG_DIR NUM_SAMPLES <<< "$JOB"

        echo "Assigning job $JOB_IDX to GPU $GPU"

        CUDA_VISIBLE_DEVICES=$GPU python src/inference/run_sampling.py \
            --job_dir "$JOB_DIR" \
            --split "$SPLIT" \
            --sampler_config_dir "$SAMPLER_CONFIG_DIR" \
            --gpu_id $GPU \
            --gpu_memory_utilization 0.8 \
            --n $NUM_SAMPLES &

        GPU_PIDS[$GPU]=$!
        ((JOB_IDX++))

        # Stop if no more jobs
        if [ "$JOB_IDX" -ge "$TOTAL_JOBS" ]; then
            break
        fi
    done

    sleep 1
done

# -----------------------------
# Step 3: Wait for remaining jobs
# -----------------------------
wait
echo "All jobs finished"