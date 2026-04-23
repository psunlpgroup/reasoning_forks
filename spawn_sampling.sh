#!/bin/bash

trap 'echo "Cleaning up..."; kill -9 -$$ 2>/dev/null' EXIT

# -----------------------------
# GPUs
# -----------------------------
AVAILABLE_GPUS=(4 5 6 7)

# -----------------------------
# Load jobs from file
# -----------------------------
JOB_FILE="job_list.txt"

mapfile -t JOB_SPLIT_PAIRS < "$JOB_FILE"

echo "Total jobs: ${#JOB_SPLIT_PAIRS[@]}"
echo "Available GPUs: ${AVAILABLE_GPUS[*]}"

# -----------------------------
# Build job queue
# -----------------------------
JOBS=()

for PAIR in "${JOB_SPLIT_PAIRS[@]}"; do
    IFS="|" read -r JOB_DIR RUN_NAME SPLIT NUM_SAMPLES <<< "$PAIR"

    echo "Processing: $JOB_DIR/$RUN_NAME"

    mapfile -t SAMPLE_DIRS < <(
        find "$JOB_DIR" -mindepth 1 -maxdepth 4 -type d \
        -path "$JOB_DIR/$RUN_NAME" | sort
    )

    for SAMPLE_DIR in "${SAMPLE_DIRS[@]}"; do
        SAMPLER_CONFIG_DIR=$(realpath --relative-to="$JOB_DIR" "$SAMPLE_DIR")

        JOBS+=("$JOB_DIR|$SPLIT|$SAMPLER_CONFIG_DIR|$NUM_SAMPLES")
    done
done

echo "Expanded jobs: ${#JOBS[@]}"

# -----------------------------
# Scheduler
# -----------------------------
declare -A GPU_PIDS

JOB_IDX=0
TOTAL_JOBS=${#JOBS[@]}

while [ "$JOB_IDX" -lt "$TOTAL_JOBS" ]; do
    for GPU in "${AVAILABLE_GPUS[@]}"; do

        # Skip busy GPU
        if [[ -n "${GPU_PIDS[$GPU]}" ]]; then
            if kill -0 "${GPU_PIDS[$GPU]}" 2>/dev/null; then
                continue
            fi
        fi

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

        if [ "$JOB_IDX" -ge "$TOTAL_JOBS" ]; then
            break
        fi
    done

    sleep 1
done

wait
echo "All jobs finished"