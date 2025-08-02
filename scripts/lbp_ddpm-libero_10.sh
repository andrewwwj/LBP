#!/bin/bash

PLANNER_DIR=$1 # Get planner directory from the first argument
PLANNER_CKPT=$2
SEED=$3
NUM_PROCS=$4
BS_PER_PROC=$5
ITER_TOTAL=$6
SAVE_INTERVAL=$7
CHUNK_LENGTH=$8
MODEL_NAME=$9
ENGINE_NAME=${10}
IMG_SIZE=${11}
USE_AC=${12}
LEARNING_RATE=${13}
WEIGHT_DECAY=${14}
ETA_MIN_LR=${15}
LOG_INTERVAL=${16}
RECURSIVE_PLANNING_STEP=${17}
DATASET_DIR=${18}
LIBERO_TASK=${19}
NUM_WORKERS=${20}
PIN_MEMORY=${21}
EXPERIMENT_NAME=${22}

BS_TOTAL=$((NUM_PROCS * BS_PER_PROC))
NUM_ITERS=$((ITER_TOTAL / BS_TOTAL))
WARM_STEPS=$((4000/${NUM_PROCS}))

#DATE=$(date +"%m%d")
PORT=29581
AVAILABLE_GPUS="0"  # "0,1"
EXPERIMENT_NAME="runnings/${DATE}_${LIBERO_TASK}_${MODEL_NAME}_hor${RECURSIVE_PLANNING_STEP}_bs$((NUM_PROCS*BS_PER_PROC))_seed${SEED}"

TRAIN_ARGS=(
    --seed $SEED
    --output_dir $EXPERIMENT_NAME
    --gpus $AVAILABLE_GPUS 
    --num_iters $NUM_ITERS
    --chunk_length $CHUNK_LENGTH 
    --model_name $MODEL_NAME 
    --engine_name $ENGINE_NAME 
    --dataset_path $DATASET_DIR/$LIBERO_TASK
    --img_size $IMG_SIZE 
    --batch_size $BS_PER_PROC
    --pin_mem $PIN_MEMORY
    --num_workers $NUM_WORKERS 
    --use_ac $USE_AC 
    --learning_rate $LEARNING_RATE 
    --weight_decay $WEIGHT_DECAY 
    --eta_min_lr $ETA_MIN_LR 
    --save_interval $SAVE_INTERVAL 
    --warm_steps $WARM_STEPS 
    --log_interval $LOG_INTERVAL 
    --recursive_step $RECURSIVE_PLANNING_STEP 
    --imaginator_ckpt_path $PLANNER_DIR/$PLANNER_CKPT
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
