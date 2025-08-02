#!/bin/bash

EXPERIMENT_NAME=$1
SEED=$2
NUM_PROCS=$3
BS_PER_PROC=$4
ITER_TOTAL=$5
SAVE_INTERVAL=$6
MODEL_NAME=$7
ENGINE_NAME=$8
IMG_SIZE=$9
LEARNING_RATE=${10}
WEIGHT_DECAY=${11}
ETA_MIN_LR=${12}
LOG_INTERVAL=${13}
RECURSIVE_STEP=${14}
REC_PLAN_COEF=${15}
DATASET_DIR=${16}
LIBERO_TASK=${17}
NUM_WORKERS=${18}
PIN_MEMORY=${19}
AVAILABLE_GPUS=${20}

NUM_ITERS=$((ITER_TOTAL / (NUM_PROCS * BS_PER_PROC)))
WARM_STEPS=$((4000/${NUM_PROCS}))

TRAIN_ARGS=(
    --seed $SEED
    --output_dir $EXPERIMENT_NAME
    --gpus $AVAILABLE_GPUS
    --num_iters $NUM_ITERS
    --model_name $MODEL_NAME
    --engine_name $ENGINE_NAME
    --dataset_path $DATASET_DIR/$LIBERO_TASK
    --img_size $IMG_SIZE
    --batch_size $BS_PER_PROC
    --pin_mem $PIN_MEMORY
    --num_workers $NUM_WORKERS
    --learning_rate $LEARNING_RATE
    --weight_decay $WEIGHT_DECAY
    --eta_min_lr $ETA_MIN_LR
    --save_interval $SAVE_INTERVAL
    --warm_steps $WARM_STEPS
    --log_interval $LOG_INTERVAL
    --recursive_step $RECURSIVE_STEP
    --rec_plan_coef $REC_PLAN_COEF
)

if [ $NUM_PROCS -eq 1 ]; then
    echo "Running with a single GPU..."
    python train_policy_sim.py "${TRAIN_ARGS[@]}"
else
    echo "Running with $NUM_PROCS GPUs using torchrun..."
    PORT=26501
    torchrun \
        --nproc_per_node=${NUM_PROCS} \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr="127.0.0.1" \
        --master_port=${PORT} \
        train_policy_sim.py "${TRAIN_ARGS[@]}"
fi
