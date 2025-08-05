#!/bin/bash
#export WANDB_START_METHOD=thread

RUN_PLANNER=1
RUN_POLICY=1

# --- Common Configuration ---
TASK_NAME="libero_10"
BASE_DIR="runnings/${TASK_NAME}/exp"
TEMP_DIR=$BASE_DIR
COUNTER=1
while [ -d "$BASE_DIR" ]; do
    BASE_DIR="${TEMP_DIR}${COUNTER}"
    COUNTER=$((COUNTER + 1))
done
DATASET_DIR="/home/andrew/pyprojects/datasets/Libero/256x256/processed"
SEED=42 #3407
NUM_WORKERS=8
PIN_MEMORY=True
NUM_PROCS=1
AVAILABLE_GPUS="0" # "0,1"
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

# --- Planner-Specific Configuration --- #
PLANNER_ITER_TOTAL=6400000
PLANNER_MODEL_NAME="mid_planner_dnce_noise"
PLANNER_RECURSIVE_STEP=4
PLANNER_REC_PLAN_COEF=0.5
PLANNER_EXP_DIR="${BASE_DIR}/$(date +"%m-%d")_${PLANNER_MODEL_NAME}_bs${BS_PER_PROC}_seed${SEED}"
#PLANNER_EXP_DIR="runnings/0731_libero_10_mid_planner_dnce_noise_bs64_seed3407"  # if choose certain checkpoint
PLANNER_CKPT="Model_ckpt_100000.pth"

# --- Policy-Specific Configuration --- #
POLICY_ITER_TOTAL=6400000
POLICY_MODEL_NAME="lbp_policy_ddpm_res34_libero"
POLICY_RECURSIVE_STEP=2
POLICY_CHUNK_LENGTH=6
POLICY_USE_AC=True
POLICY_EXP_DIR="${BASE_DIR}/$(date +"%m-%d")_${POLICY_MODEL_NAME}_hor${POLICY_RECURSIVE_STEP}_bs${BS_PER_PROC}_seed${SEED}"

# --- Execution --- #
if [ $RUN_PLANNER = 1 ]; then
  echo "======================================================"
  echo "Running planner script..."
  echo "Planner Checkpoint: ${PLANNER_EXP_DIR}"
  echo "======================================================"
  bash scripts/planner_libero.sh \
      "${PLANNER_EXP_DIR}" \
      "${SEED}" \
      "${NUM_PROCS}" \
      "${BS_PER_PROC}" \
      "${PLANNER_ITER_TOTAL}" \
      "${SAVE_INTERVAL}" \
      "${PLANNER_MODEL_NAME}" \
      "${ENGINE_NAME}" \
      "${IMG_SIZE}" \
      "${LEARNING_RATE}" \
      "${WEIGHT_DECAY}" \
      "${ETA_MIN_LR}" \
      "${LOG_INTERVAL}" \
      "${WARM_STEPS}" \
      "${PLANNER_RECURSIVE_STEP}" \
      "${PLANNER_REC_PLAN_COEF}" \
      "${DATASET_DIR}" \
      "${TASK_NAME}" \
      "${NUM_WORKERS}" \
      "${PIN_MEMORY}" \
      "${AVAILABLE_GPUS}"

  if [ $? -ne 0 ]; then
      echo "Planner script failed. Aborting."
      exit 1
  fi
fi

if [ $RUN_POLICY = 1 ]; then
  echo "======================================================"
  echo "Running policy script..."
  echo "Planner Checkpoint: ${PLANNER_EXP_DIR}/${PLANNER_CKPT}"
  echo "Policy Checkpoint: ${POLICY_EXP_DIR}"
  echo "======================================================"
  bash scripts/lbp_ddpm-libero_10.sh \
      "${PLANNER_EXP_DIR}" \
      "${PLANNER_CKPT}" \
      "${SEED}" \
      "${NUM_PROCS}" \
      "${BS_PER_PROC}" \
      "${POLICY_ITER_TOTAL}" \
      "${SAVE_INTERVAL}" \
      "${POLICY_CHUNK_LENGTH}" \
      "${POLICY_MODEL_NAME}" \
      "${ENGINE_NAME}" \
      "${IMG_SIZE}" \
      "${POLICY_USE_AC}" \
      "${LEARNING_RATE}" \
      "${WEIGHT_DECAY}" \
      "${ETA_MIN_LR}" \
      "${LOG_INTERVAL}" \
      "${WARM_STEPS}" \
      "${POLICY_RECURSIVE_STEP}" \
      "${DATASET_DIR}" \
      "${TASK_NAME}" \
      "${NUM_WORKERS}" \
      "${PIN_MEMORY}" \
      "${POLICY_EXP_DIR}" \
      "${AVAILABLE_GPUS}"

  if [ $? -ne 0 ]; then
      echo "Policy script failed."
      exit 1
  fi
fi

echo "======================================================"
echo "All scripts finished successfully."
echo "======================================================"