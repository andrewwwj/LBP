#!/bin/bash
export MUJOCO_GL="egl"

RUN_PLANNER=1
RUN_POLICY=1

# --- Common Configuration ---
TASK_NAME="libero_10"
BASE_DIR="logs/${TASK_NAME}/exp"
TEMP_DIR=$BASE_DIR
COUNTER=1
while [ -d "$BASE_DIR" ]; do
    BASE_DIR="${TEMP_DIR}${COUNTER}"
    COUNTER=$((COUNTER + 1))
done
mkdir -p $BASE_DIR
touch "$BASE_DIR/train_info.txt"

DATASET_DIR="/home/andrew/pyprojects/datasets/${TASK_NAME}"

SEED=3407
NUM_WORKERS=8
PIN_MEMORY=True
NUM_PROCS=1
AVAILABLE_GPUS="0"
BS_TOTAL=64
BS_PER_PROC=$((BS_TOTAL / NUM_PROCS))
SAVE_INTERVAL=10000
LEARNING_RATE=3e-4
WEIGHT_DECAY=0
ETA_MIN_LR=0
LOG_INTERVAL=100
WARM_STEPS=2000
IMG_SIZE=224
ENGINE_NAME="build_libero_engine"

# --- Planner-Specific Configuration ---
PLANNER_ITER_TOTAL=6400000
PLANNER_ITER=$((PLANNER_ITER_TOTAL / BS_TOTAL))
PLANNER_MODEL_NAME="mid_planner_dnce_noise"
PLANNER_RECURSIVE_STEP=4
PLANNER_REC_PLAN_COEF=0.5
PLANNER_EXP_DIR="${BASE_DIR}/$(date +"%m-%d")_${PLANNER_MODEL_NAME}_hor${PLANNER_RECURSIVE_STEP}_bs${BS_PER_PROC}_seed${SEED}"
#PLANNER_EXP_DIR="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/baseline/08-11_mid_planner_dnce_noise_bs64_seed3407"
PLANNER_CKPT="Model_ckpt_100000.pth"

# --- Policy-Specific Configuration ---
POLICY_ITER_TOTAL=6400000
POLICY_ITER=$((POLICY_ITER_TOTAL / BS_TOTAL))
POLICY_MODEL_NAME="lbp_policy_ddpm_res34_libero"
POLICY_RECURSIVE_STEP=2
CHUNK_LENGTH=6
USE_AC=True
POLICY_EXP_DIR="${BASE_DIR}/$(date +"%m-%d")_${POLICY_MODEL_NAME}_hor${POLICY_RECURSIVE_STEP}_bs${BS_PER_PROC}_seed${SEED}"
#POLICY_EXP_DIR="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/exp24/08-09_lbp_policy_ddpm_res34_libero_hor2_bs64_seed42"
# --- Execution ---
# 1) Train planner
if [ $RUN_PLANNER = 1 ]; then
  echo
  echo "======================================================"
  echo "Running planner script..."
  echo "Planner Checkpoint: ${PLANNER_EXP_DIR}"
  echo "======================================================"
#  PORT=26501
#  torchrun \
#      --nproc_per_node=${NUM_PROCS} \
#      --nnodes=1 \
#      --node_rank=0 \
#      --master_addr="127.0.0.1" \
#      --master_port=${PORT} \
  python train_policy_sim.py \
        --compile \
        --seed $SEED \
        --output_dir "$PLANNER_EXP_DIR" \
        --gpus $AVAILABLE_GPUS \
        --num_iters $PLANNER_ITER \
        --model_name $PLANNER_MODEL_NAME \
        --engine_name $ENGINE_NAME \
        --dataset_path $DATASET_DIR \
        --img_size $IMG_SIZE \
        --batch_size $BS_PER_PROC \
        --pin_mem $PIN_MEMORY \
        --num_workers $NUM_WORKERS \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --eta_min_lr $ETA_MIN_LR \
        --save_interval $SAVE_INTERVAL \
        --warm_steps $WARM_STEPS \
        --log_interval $LOG_INTERVAL \
        --recursive_step $PLANNER_RECURSIVE_STEP \
        --rec_plan_coef $PLANNER_REC_PLAN_COEF

  if [ $? -ne 0 ]; then
      echo "Planner script failed. Aborting."
      exit 1
  fi
  echo "Planner script finished. Results saved to $LOG_DIR"
fi

# 2) Train Policy
if [ $RUN_POLICY = 1 ]; then
  echo
  echo "======================================================"
  echo "Running policy script..."
  echo "Planner Checkpoint: ${PLANNER_EXP_DIR}/${PLANNER_CKPT}"
  echo "Policy Checkpoint: ${POLICY_EXP_DIR}"
  echo "======================================================"
#  PORT=26501
#  torchrun \
#      --nproc_per_node=${NUM_PROCS} \
#      --nnodes=1 \
#      --node_rank=0 \
#      --master_addr="127.0.0.1" \
#      --master_port=${PORT} \
  python train_policy_sim.py \
        --compile \
        --seed $SEED \
        --output_dir "$POLICY_EXP_DIR" \
        --gpus $AVAILABLE_GPUS \
        --num_iters $POLICY_ITER \
        --chunk_length $CHUNK_LENGTH \
        --model_name $POLICY_MODEL_NAME \
        --engine_name $ENGINE_NAME \
        --dataset_path $DATASET_DIR \
        --img_size $IMG_SIZE \
        --batch_size $BS_PER_PROC \
        --pin_mem $PIN_MEMORY \
        --num_workers $NUM_WORKERS \
        --use_ac $USE_AC \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --eta_min_lr $ETA_MIN_LR \
        --save_interval $SAVE_INTERVAL \
        --warm_steps $WARM_STEPS \
        --log_interval $LOG_INTERVAL \
        --recursive_step $POLICY_RECURSIVE_STEP \
        --imaginator_ckpt_path "$PLANNER_EXP_DIR/$PLANNER_CKPT"
  if [ $? -ne 0 ]; then
      echo "Policy script failed."
      exit 1
  fi
  echo "Policy script finished. Results saved to $BASE_DIR"
fi

echo "======================================================"
echo "All scripts finished successfully."
echo "======================================================"