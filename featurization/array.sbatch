#!/bin/sh
#SBATCH --job-name arrayr2
#SBATCH --partition commons
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --output=log%a.txt
#SBATCH --array=0-99

#python several_mutate.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID
#python -u create_dists_parts.py $SLURM_ARRAY_TASK_ID
python -u create_dataset_parts.py $SLURM_ARRAY_TASK_ID r2
