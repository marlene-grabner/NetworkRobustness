#!/bin/bash
#SBATCH --job-name=global_efficiency_baseline
#SBATCH --output=outputs/logs/global_efficiency_baseline_%A_%a.out
#SBATCH --error=outputs/logs/global_efficiency_baseline_%A_%a.err
#SBATCH --array=1-16
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_80

# Run scripts/global_properties/global_efficiency/generate_baseline_job_table.py
# once beforehand to (re)create
# slurm_scripts/global_properties/global_efficiency/jobs/baseline_jobs.csv and
# print the correct --array range (16 with the current 4 base networks + 12
# null models).

set -euo pipefail

uv run scripts/global_properties/global_efficiency/run_baseline_task.py --job-row "${SLURM_ARRAY_TASK_ID}"
