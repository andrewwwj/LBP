#!/bin/bash
export MUJOCO_GL="glx"  # handle issue AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
AVAILABLE_GPUS="0"
TASK_NAME="libero_10" # ["libero_goal", "libero_spatial", "libero_10"]
DATA_PATH="/home/andrew/pyprojects/datasets/libero_10"

# --- Libero-10 ---
#CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/baseline/08-11_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407"  # Baseline
#CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/baseline_proprio/08-22_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407"  # Baseline w/ proprio+goal
#CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/baseline_wo_goal/"  # Baseline w/o goal
#CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/cfg/08-18_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407"  # CFG
# --- Libero-10-wo-task8 ---
#CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10_wo_task8/baseline/08-27_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407" # baseline
#CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10_wo_task8/cfg_w_vg_mask/08-31_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407"  # CFG w/ vpg + vg masking
#CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10_wo_task8/exp1/08-31_energy_guided_planner_bs64_seed3407"  # energy w/ p anchor + vg energy
# CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10_wo_task8/exp3/09-03_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407"  # CFG w/ randn embedding
#CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10_wo_task8/cfg_mask_vg/08-31_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407"  # CFG w/ mask vg only
CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10_wo_task8/exp2/09-11_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407"

echo "Start Evaluation..."
echo
echo "======================================================"
echo "Checkpoint Path: ${CKPT_PATH}"
echo "GPU: ${AVAILABLE_GPUS}"
echo "Task Name: ${TASK_NAME}"
echo "======================================================"
#W_CFG_LIST=(0 1.0 2.0 3.0)
EVAL_NUM=1
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
