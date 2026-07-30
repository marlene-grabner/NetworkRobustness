#!/bin/bash
#SBATCH --job-name=comm_fast_algos_small
#SBATCH --array=1-13%3
#SBATCH --cpus-per-task=6               
#SBATCH --mem=12G                        
#SBATCH --time=01:00:00                 
#SBATCH --output=outputs/logs/comm_fast_algos_small_%A_%a.out
#SBATCH --error=outputs/logs/comm_fast_algos_small_%A_%a.err

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm_scripts/local_structure/tmp/tasks_fast_algos_small.txt)
PARQUET=$(echo "$LINE" | cut -f1)
BASELINE=$(echo "$LINE" | cut -f2)
OUT=$(echo "$LINE" | cut -f3)
ALGO=$(echo "$LINE" | cut -f4)
        
uv run notebooks_general_analysis/local_structure/similarity_perturbation/test_community_pertrubed.py "$PARQUET" "$BASELINE" "$OUT" "$ALGO" --n-jobs 6
