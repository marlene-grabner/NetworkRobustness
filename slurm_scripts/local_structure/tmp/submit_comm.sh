#!/bin/bash
#SBATCH --job-name=comm_det
#SBATCH --array=1-4%5
#SBATCH --cpus-per-task=6               
#SBATCH --mem=32G                        
#SBATCH --time=05:00:00                 
#SBATCH --output=outputs/logs/comm_%A_%a.out
#SBATCH --error=outputs/logs/comm_%A_%a.err

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm_scripts/local_structure/tmp/community_tasks_man.txt)
PARQUET=$(echo "$LINE" | cut -f1)
BASELINE=$(echo "$LINE" | cut -f2)
OUT=$(echo "$LINE" | cut -f3)
ALGO=$(echo "$LINE" | cut -f4)

# Pass the --n-jobs 6 flag to your worker script to use the SLURM allocated CPUs
uv run notebooks_general_analysis/local_structure/similarity_perturbation/test_community_pertrubed.py "$PARQUET" "$BASELINE" "$OUT" "$ALGO" --n-jobs 6
