#!/bin/bash

# --- Common Configuration ---
SEED=3407
NUM_WORKERS=8
PIN_MEMORY=True
NUM_PROCS=1
AVAILABLE_GPUS="0" # "0,1"
BS_PER_PROC=64
BS_TOTAL=$((NUM_PROCS * BS_PER_PROC))
ITER_TOTAL=6400000
SAVE_INTERVAL=10000
LEARNING_RATE=3e-4
WEIGHT_DECAY=0
ETA_MIN_LR=0
LOG_INTERVAL=100
IMG_SIZE=224
ENGINE_NAME="build_libero_engine"

# --- Planner-Specific Configuration ---
PLANNER_MODEL_NAME="mid_planner_dnce_noise"
PLANNER_RECURSIVE_STEP=4
PLANNER_REC_PLAN_COEF=0.5

# --- Policy-Specific Configuration ---
POLICY_MODEL_NAME="lbp_policy_ddpm_res34_libero"
POLICY_RECURSIVE_STEP=2
POLICY_CHUNK_LENGTH=6
POLICY_USE_AC=True

# --- Experiment Setup ---
DATE=$(date +"%m%d")
LIBERO_TASK="libero_10"
BASE_DIR="runnings/${LIBERO_TASK}/${DATE}"
DATASET_DIR="/home/andrew/pyprojects/datasets/Libero/256x256/processed"
PLANNER_EXP_DIR="${BASE_DIR}_${PLANNER_MODEL_NAME}_bs$(BS_TOTAL)_seed${SEED}"
PLANNER_DIR=$PLANNER_EXP_DIR
COUNTER=1
while [ -d "$PLANNER_EXP_DIR" ]; do
    PLANNER_EXP_DIR="${PLANNER_DIR}_exp${COUNTER}"
    COUNTER=$((COUNTER + 1))
done
PLANNER_CKPT="Model_ckpt_100000.pth"
POLICY_EXP_DIR="${BASE_DIR}_${POLICY_MODEL_NAME}_hor${POLICY_RECURSIVE_STEP}_bs$(BS_TOTAL)_seed${SEED}"

# --- Script Execution ---
echo "======================================================"
echo "Running planner script..."
echo "======================================================"
bash scripts/planner_libero.sh \
    "${PLANNER_EXP_DIR}" \
    "${SEED}" \
    "${NUM_PROCS}" \
    "${BS_PER_PROC}" \
    "${ITER_TOTAL}" \
    "${SAVE_INTERVAL}" \
    "${PLANNER_MODEL_NAME}" \
    "${ENGINE_NAME}" \
    "${IMG_SIZE}" \
    "${LEARNING_RATE}" \
    "${WEIGHT_DECAY}" \
    "${ETA_MIN_LR}" \
    "${LOG_INTERVAL}" \
    "${PLANNER_RECURSIVE_STEP}" \
    "${PLANNER_REC_PLAN_COEF}" \
    "${DATASET_DIR}" \
    "${LIBERO_TASK}" \
    "${NUM_WORKERS}" \
    "${PIN_MEMORY}" \
    "${AVAILABLE_GPUS}"

# Check if the planner script was successful before proceeding
if [ $? -ne 0 ]; then
    echo "Planner script failed. Aborting."
    exit 1
fi

echo "======================================================"
echo "Running policy script..."
echo "Planner Checkpoint: ${PLANNER_EXP_DIR}/${PLANNER_CKPT}"
echo "======================================================"
bash scripts/lbp_ddpm-libero_10.sh \
    "${PLANNER_EXP_DIR}" \
    "${PLANNER_CKPT}" \
    "${SEED}" \
    "${NUM_PROCS}" \
    "${BS_PER_PROC}" \
    "${ITER_TOTAL}" \
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
    "${POLICY_RECURSIVE_STEP}" \
    "${DATASET_DIR}" \
    "${LIBERO_TASK}" \
    "${NUM_WORKERS}" \
    "${PIN_MEMORY}" \
    "${POLICY_EXP_DIR}" \
    "${AVAILABLE_GPUS}"

if [ $? -ne 0 ]; then
    echo "Policy script failed."
    exit 1
fi

echo "======================================================"
echo "All scripts finished successfully."
echo "======================================================"