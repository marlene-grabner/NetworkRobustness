#!/bin/bash
#SBATCH --job-name=baseline_rank
#SBATCH --output=outputs/logs/baseline_%A_%a.out
#SBATCH --error=outputs/logs/baseline_%A_%a.err
#SBATCH --array=1-64
#SBATCH --time=00:30:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1

# Run notebooks/generate_baseline_job_table.py once beforehand to (re)create
# results/job_tables/baseline_jobs.csv and print the correct --array range.

set -euo pipefail

uv run scripts/seed_expansion/expansion/run_baseline_task.py --job-row "$SLURM_ARRAY_TASK_ID"