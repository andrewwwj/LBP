#!/bin/bash
export WANDB_API_KEY=a9abdbf33f19f62cbbd321e4498210cbaaf1efc0

# fix
SEED=42
NUM_PROCS=2
BS_PER_PROC=32
CURRENT_DATE="0223"
LIBERO_SUBSUITE="libero_10"

# hyper
PORT=29501
AVAILABLE_GPUS="2,3"
MODEL_NAME="bc_policy_res34_libero"
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
        --num_iters 100000 \
        --chunk_length 6 \
        --model_name $MODEL_NAME \
        --engine_name build_libero_engine \
        --dataset_path /dysData/nhy/datasets/libero/$LIBERO_SUBSUITE \
        --img_size 256 \
        --batch_size $BS_PER_PROC \
        --num_workers 8 \
        --use_ac True \
        --learning_rate 3e-4 \
        --weight_decay 0 \
        --eta_min_lr 0 \
        --save_interval 20000 \
        --warm_steps 2000 \
        --log_interval 50 \
