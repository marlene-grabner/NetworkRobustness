#!/bin/bash
#SBATCH --job-name=perturbed_metrics
#SBATCH --output=outputs/logs/perturbed_%A_%a.out
#SBATCH --error=outputs/logs/perturbed_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=1-918%50
#SBATCH --mem=8G
#SBATCH --mail-type=END,TIME_LIMIT_80,TIME_LIMIT

#SBATCH --cpus-per-task=1

# MUST run after the baseline array has fully completed (it reads
# outputs/seed_expansion/expansion/baseline_rankings/ and outputs/seed_expansion/expansion/node_index/ written by that step).
# Submit with a dependency, e.g.:
#   BASE_JOBID=$(sbatch --parsable slurm/baseline_array.sbatch)
#   sbatch --dependency=afterok:${BASE_JOBID} slurm/perturbed_array.sbatch

set -euo pipefail

uv run scripts/seed_expansion/expansion/run_perturbed_task.py --job-row "${SLURM_ARRAY_TASK_ID}" --k-list 10,25,50,100