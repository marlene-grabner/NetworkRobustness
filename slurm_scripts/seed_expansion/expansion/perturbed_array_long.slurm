#!/bin/bash
#SBATCH --job-name=perturbed_metrics
#SBATCH --output=outputs/logs/perturbed_%A_%a.out
#SBATCH --error=outputs/logs/perturbed_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --array=1-31%15
#SBATCH --mem=8G
#SBATCH --mail-type=END,TIME_LIMIT_80,TIME_LIMIT

#SBATCH --cpus-per-task=1

# MUST run after the baseline array has fully completed (it reads
# outputs/seed_expansion/expansion/baseline_rankings/ and outputs/seed_expansion/expansion/node_index/ written by that step).
# Submit with a dependency, e.g.:
#   BASE_JOBID=$(sbatch --parsable slurm/baseline_array.sbatch)
#   sbatch --dependency=afterok:${BASE_JOBID} slurm/perturbed_array.sbatch

set -euo pipefail


# 1. Define the path here so it's easy to change later
JOB_CSV="slurm_scripts/seed_expansion/expansion/jobs/leftover_jobs.csv"

# 2. Pass it to the python script using backslashes to keep it readable
uv run scripts/seed_expansion/expansion/run_perturbed_task.py \
    --job-table "${JOB_CSV}" \
    --job-row "${SLURM_ARRAY_TASK_ID}" \
    --k-list 10,25,50,100
