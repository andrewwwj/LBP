#!/bin/bash
export MUJOCO_GL="glx"  # handle issue AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
AVAILABLE_GPUS="0"
TASK_NAME="libero_10" # ["libero_goal", "libero_spatial", "libero_10"]
CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/exp24/08-09_lbp_policy_ddpm_res34_libero_hor2_bs64_seed42"
echo "======================================================"
echo "Starting Evaluation"
echo "======================================================"
echo "Checkpoint Path: ${CKPT_PATH}"
echo "GPU: ${AVAILABLE_GPUS}"
echo "Task Name: ${TASK_NAME}"
echo "------------------------------------------------------"
for i in {1..3}; do
  echo "--- Evaluation run $i of $EVAL_NUM ---"
  python3 eval_libero.py \
      --compile \
      --gpu ${AVAILABLE_GPUS} \
      --ckpt_path ${CKPT_PATH} \
      --task_name ${TASK_NAME}
done
echo "======================================================"
echo "Evaluation Finished"
echo "======================================================"
