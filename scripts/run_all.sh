#!/bin/bash
export MUJOCO_GL="egl"

RUN_PLANNER=0
RUN_POLICY=1

# --- Common Configuration ---
TASK_NAME="libero_10_wo_task8"
LOG_DIR="logs/${TASK_NAME}/exp"
TEMP_DIR=$LOG_DIR
COUNTER=1
while [ -d "$LOG_DIR" ]; do
    LOG_DIR="${TEMP_DIR}${COUNTER}"
    COUNTER=$((COUNTER + 1))
done

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
#PLANNER_EXP_DIR="${LOG_DIR}/$(date +"%m-%d")_${PLANNER_MODEL_NAME}_hor${PLANNER_RECURSIVE_STEP}_bs${BS_PER_PROC}_seed${SEED}"
#PLANNER_EXP_DIR="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10/baseline/08-11_mid_planner_dnce_noise_bs64_seed3407"  # libero10 baseline
PLANNER_EXP_DIR="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10_wo_task8/baseline_w_p/08-29_mid_planner_dnce_noise_hor4_bs64_seed3407"  # # libero10-wo-t8 baseline
PLANNER_CKPT_PATH="${PLANNER_EXP_DIR}/Model_ckpt_100000.pth"

# --- Policy-Specific Configuration ---
POLICY_ITER_TOTAL=6400000
POLICY_ITER=$((POLICY_ITER_TOTAL / BS_TOTAL))
POLICY_MODEL_NAME="lbp_policy_ddpm_res34_libero"

POLICY_RECURSIVE_STEP=2
CHUNK_LENGTH=6
USE_AC=True
POLICY_GUIDANCE="cfg"

DIFFUSION_INPUT_KEY="vg"
ENERGY_INPUT_KEY=""

POLICY_EXP_DIR="${LOG_DIR}/$(date +"%m-%d")_${POLICY_MODEL_NAME}_hor${POLICY_RECURSIVE_STEP}_bs${BS_PER_PROC}_seed${SEED}"
POLICY_CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10_wo_task8/baseline_w_p/08-29_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407/Model_ckpt_100000.pth"
EXPERT_POLICY_CKPT_PATH="/home/andrew/pyprojects/GenerativeRL/LBP/logs/libero_10_wo_task8/baseline/08-27_lbp_policy_ddpm_res34_libero_hor2_bs64_seed3407/Model_ckpt_100000.pth"
if [ -n "$EXPERT_POLICY_CKPT_PATH" ]; then
  POLICY_EXP_DIR="${LOG_DIR}/$(date +"%m-%d")_energy_guided_planner_bs${BS_PER_PROC}_seed${SEED}"
  POLICY_ARG="--policy_ckpt_path $POLICY_CKPT_PATH --expert_policy_ckpt_path $EXPERT_POLICY_CKPT_PATH"
elif [ -n "$POLICY_CKPT_PATH" ]; then
  POLICY_ARG="--policy_ckpt_path $POLICY_CKPT_PATH"
else
  POLICY_ARG=""
fi

# Set up log directory
mkdir -p $LOG_DIR
cp "./models/components/ActionHead.py" "$LOG_DIR/"
cp "./models/MidPlanner.py" "$LOG_DIR/"
cp "./models/LBP.py" "$LOG_DIR/"
touch "$LOG_DIR/train_info.txt"

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
  echo "Guidance: ${POLICY_GUIDANCE}"
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
        --imaginator_ckpt_path $PLANNER_CKPT_PATH \
        --guidance_mode $POLICY_GUIDANCE \
        --diffusion_input_key $DIFFUSION_INPUT_KEY \
        --energy_input_key $ENERGY_INPUT_KEY \
        $POLICY_ARG
  if [ $? -ne 0 ]; then
      echo "Policy script failed."
      exit 1
  fi
  echo "Policy script finished. Results saved to $LOG_DIR"
fi

echo "======================================================"
echo "All scripts finished successfully."
echo "======================================================"