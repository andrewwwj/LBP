#!/bin/bash
export MUJOCO_GL="glx"  # handle issue AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
AVAILABLE_GPUS="0"
TASK_NAME="libero_10" # ["libero_goal", "libero_spatial", "libero_10"]
DATA_PATH="/home/andrew/pyprojects/datasets/libero_10"
CKPT_PATH="/mnt/nas3/andrew/projects/LBP/logs/server_25/exp2/libero_10/08-18_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407"
echo "======================================================"
echo "Starting Evaluation"
echo "======================================================"
echo "Checkpoint Path: ${CKPT_PATH}"
echo "GPU: ${AVAILABLE_GPUS}"
echo "Task Name: ${TASK_NAME}"
echo "------------------------------------------------------"

EVAL_NUM=5
for i in $(seq 1 $EVAL_NUM); do
  echo "--- Evaluation run $i of $EVAL_NUM ---"
  python3 eval_libero.py \
      --compile \
      --gpu $AVAILABLE_GPUS \
      --dataset_path $DATA_PATH \
      --ckpt_path $CKPT_PATH \
      --task_name $TASK_NAME
done
echo "======================================================"
echo "Evaluation Finished"
echo "======================================================"
