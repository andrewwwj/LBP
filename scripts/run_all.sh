#!/bin/bash

# This script runs the planner and then the LBP policy.
# It defines a shared experiment directory and configuration to ensure
# the policy script can find and use the trained planner model.

# --- Shared Configuration ---
SEED=3407
NUM_WORKERS=8
PIN_MEMORY=True
NUM_PROCS=1
BS_PER_PROC=64
ITER_TOTAL=6400000
SAVE_INTERVAL=10000
LEARNING_RATE=3e-4
WEIGHT_DECAY=0
ETA_MIN_LR=0
LOG_INTERVAL=100
IMG_SIZE=224

# --- Common Configuration ---
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
DATASET_DIR="/home/andrew/pyprojects/datasets/Libero/256x256/processed"
EXP_DIR="runnings/${DATE}_${LIBERO_TASK}_${PLANNER_MODEL_NAME}_bs$((NUM_PROCS*BS_PER_PROC))_seed${SEED}"
PLANNER_CKPT="Model_ckpt_100000.pth"
POLICY_EXP_DIR="runnings/${DATE}_${LIBERO_TASK}_${MODEL_NAME}_hor${RECURSIVE_PLANNING_STEP}_bs$((NUM_PROCS*BS_PER_PROC))_seed${SEED}"

# --- Script Execution ---
echo "Running planner script..."
echo "Experiment directory: ${EXP_DIR}"
bash scripts/planner_libero.sh \
    "${EXP_DIR}" \
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
    "${PIN_MEMORY}"

# Check if the planner script was successful before proceeding
if [ $? -ne 0 ]; then
    echo "Planner script failed. Aborting."
    exit 1
fi

echo "Running policy script..."
bash scripts/lbp_ddpm-libero_10.sh \
    "${EXPERIMENT_DIR}" \
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
    "${POLICY_EXP_DIR}"

if [ $? -ne 0 ]; then
    echo "Policy script failed."
    exit 1
fi

echo "All scripts finished successfully."