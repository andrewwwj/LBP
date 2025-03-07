#!/bin/bash
export WANDB_API_KEY=a9abdbf33f19f62cbbd321e4498210cbaaf1efc0

# fix
SEED=42
NUM_PROCS=1
BS_PER_PROC=64
CURRENT_DATE="0308"
LIBERO_SUBSUITE="libero_10"

# hyper
PORT=29500
AVAILABLE_GPUS="5"
MODEL_NAME="bc_policy_ddpm_res34_libero"
EXPERIMENT_NAME="runnings/${CURRENT_DATE}-${MODEL_NAME}-${LIBERO_SUBSUITE}-bs_$((NUM_PROCS*BS_PER_PROC))-seed_${SEED}"

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
        --num_iters 200000 \
        --chunk_length 6 \
        --model_name $MODEL_NAME \
        --engine_name build_libero_engine \
        --dataset_path /mnt/ssd0/data/libero/$LIBERO_SUBSUITE \
        --img_size 224 \
        --batch_size $BS_PER_PROC \
        --num_workers 8 \
        --use_ac True \
        --learning_rate 3e-4 \
        --weight_decay 0 \
        --eta_min_lr 0 \
        --save_interval 50000 \
        --warm_steps 2000 \
        --log_interval 50 \
