MODEL_NAME=$1
NPROC=$2
PER_DEVICE_BATCH_SIZE=16

RUN_ID=${MODEL_NAME}-grpo
DATA_PATH=datasets/arithchain_2_10/train_rlvr.parquet
DATA_SIZE=1600
MAX_ITER=1600
SAVE_STEPS=100
TEST_FREQ=100
RUN_DIR=runs

python src/training/nano_r1_fsdp.py --nproc $NPROC --model_name $MODEL_NAME --run_id $RUN_ID --save_steps $SAVE_STEPS --test_freq $TEST_FREQ --run_dir $RUN_DIR --data_path $DATA_PATH --data_size $DATA_SIZE --per_device_batch_size $PER_DEVICE_BATCH_SIZE --max_iter $MAX_ITER