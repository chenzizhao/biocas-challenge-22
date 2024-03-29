#!/bin/bash
#SBATCH --array=1-1
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH -c 32
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
base_cmd=$1 #e.g. python3 train_agent.py --algo ppo --env Letter-7x7-v2 --save-interval 100 --frames 1000000000 --ltl-sampler UntilTasks_3_3_1_1 --lr 0.0001
eval "$base_cmd --wandb True --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}"
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE
