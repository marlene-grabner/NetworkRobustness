#!/bin/bash
#SBATCH --job-name=comm_infomap_large
#SBATCH --array=1-315%12
#SBATCH --cpus-per-task=8               
#SBATCH --mem=16G                        
#SBATCH --time=08:00:00                 
#SBATCH --output=outputs/logs/comm_infomap_large_%A_%a.out
#SBATCH --error=outputs/logs/comm_infomap_large_%A_%a.err

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm_scripts/local_structure/tmp/tasks_infomap_large.txt)
PARQUET=$(echo "$LINE" | cut -f1)
BASELINE=$(echo "$LINE" | cut -f2)
OUT=$(echo "$LINE" | cut -f3)
ALGO=$(echo "$LINE" | cut -f4)
        
uv run notebooks_general_analysis/local_structure/similarity_perturbation/test_community_pertrubed.py "$PARQUET" "$BASELINE" "$OUT" "$ALGO" --n-jobs 8
