#!/bin/bash
export WANDB_API_KEY=a9abdbf33f19f62cbbd321e4498210cbaaf1efc0

# fix
SEED=42
NUM_PROCS=2
BS_PER_PROC=32
CURRENT_DATE="0307"

# hyper
PORT=26501
AVAILABLE_GPUS="4,5"
MODEL_NAME="mid_planner_dnce_noise"
EXPERIMENT_NAME="runnings/${CURRENT_DATE}-mid_planner_libero_dnce-bs_$((NUM_PROCS*BS_PER_PROC))-seed_${SEED}"

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROCS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=${PORT} \
    train_policy_sim.py \
        --seed $SEED \
        --output_dir $EXPERIMENT_NAME \
        --gpus $AVAILABLE_GPUS \
        --num_iters 100000 \
        --model_name $MODEL_NAME \
        --engine_name build_libero_engine \
        --dataset_path /mnt/ssd0/data/libero \
        --img_size 224 \
        --batch_size $BS_PER_PROC \
        --num_workers 8 \
        --learning_rate 3e-4 \
        --weight_decay 0 \
        --eta_min_lr 0 \
        --save_interval 20000 \
        --warm_steps 2000 \
        --log_interval 50 \
        --recursive_step 4 \
        --rec_plan_coef 0.5
