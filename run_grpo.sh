NPROC=$1

DATA_PATH=datasets/arithchain_2_10/train_rlvr.parquet
DATA_SIZE=1600 # Train 1600
MAX_ITER=1600 # 16 epochs
MODEL_NAME=runs/reasoning_forks_sft/qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1/checkpoint-200 # SFT ep 1
RUN_ID=qwen2.5_0.5b_sft_arithchain_2_10_forward_lr1e-5_bs32_ga1-grpo
SAVE_STEPS=100
TEST_FREQ=100
RUN_DIR=runs
PER_DEVICE_BATCH_SIZE=16

python src/training/nano_r1_fsdp.py --nproc $NPROC --model_name $MODEL_NAME --run_id $RUN_ID --save_steps $SAVE_STEPS --test_freq $TEST_FREQ --run_dir $RUN_DIR --data_path $DATA_PATH --data_size $DATA_SIZE --per_device_batch_size $PER_DEVICE_BATCH_SIZE --max_iter $MAX_ITER