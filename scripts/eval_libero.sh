#!/bin/bash
export MUJOCO_GL="glx"  # handle issue AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
# -- Name of the checkpoint directory to evaluate --
# This should be the name of the folder in 'runnings' that contains your model checkpoints.
EXPERIMENT_NAME='0801_libero_10_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407'
AVAILABLE_GPUS="0"
# Specify the task suites you want to evaluate on (e.g., 'libero_10', 'libero_90').
TASK_NAME="libero_10" # ["libero_goal", "libero_spatial", "libero_10"]
#BASE_DIR="/home/andrew/pyprojects"
CKPT_PATH="runnings/${EXPERIMENT_NAME}"

echo "======================================================"
echo "Starting Evaluation"
echo "======================================================"

echo "Checkpoint Path: ${CKPT_PATH}"
echo "GPU: ${AVAILABLE_GPUS}"
echo "Task Name: ${TASK_NAME}"
echo "------------------------------------------------------"

python3 eval_libero.py \
    --gpu ${AVAILABLE_GPUS} \
    --ckpt_path ${CKPT_PATH} \
    --task_name ${TASK_NAME}

echo "======================================================"
echo "Evaluation Finished"
echo "======================================================"
