#!/bin/bash
#SBATCH --array=1-4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --partition=t4v2
#SBATCH --qos=high
base_cmd=$1 #e.g. python3 train_agent.py --algo ppo --env Letter-7x7-v2 --save-interval 100 --frames 1000000000 --ltl-sampler UntilTasks_3_3_1_1 --lr 0.0001
tasklevel=$SLURM_ARRAY_TASK_ID
task=$(($tasklevel/2))
level=$(($tasklevel%2+1))
#echo "$base_cmd -c config_task$task$level.json"
eval "$base_cmd -c config_task$task$level.json --wandb True --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}"
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE
