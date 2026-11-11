#!/bin/bash
#SBATCH --job-name=communities
#SBATCH --array=0-199%30
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_80
#SBATCH --output=outputs/logs/%A_%a.out
#SBATCH --error=outputs/logs/err_%A_%a.txt

# Community detection on all perturbed networks. The 4,860 perturbed files are
# spread over the array tasks (task i takes every 200th file), so every task
# gets a similar mix of small and large networks.
#
# Resources (estimate from local timings, cluster assumed 3x slower): about
# 1,500 CPU hours in total = about 2 h per task on 4 CPUs. A task that gets
# several astro null model files takes longer, hence 8 h.
#
# Finished files are skipped, so tasks that timed out can just be resubmitted.
# N_TASKS must stay the same as the array size.
#
# Submit from the repository root, the baselines first:
#   mkdir -p outputs/logs
#   jid=$(sbatch --parsable --array=0 slurm_scripts/local_structure/communities.slurm baselines)
#   sbatch --dependency=afterok:${jid} slurm_scripts/local_structure/communities.slurm
# Afterwards: uv run python scripts/local_structure/aggregate_communities.py

set -euo pipefail
N_TASKS=200

# Python environment on the node-local disk (see generating_perturbed_networks.slurm)
ENV_DIR="${TMPDIR:-/tmp}/noise_env_${SLURM_JOB_ID}"
trap 'rm -rf "${ENV_DIR}"' EXIT
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
export UV_LINK_MODE=copy
uv sync --frozen --no-dev --quiet
PYTHON="${ENV_DIR}/bin/python"

# every worker is one process; no extra threads from numpy or igraph
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Array task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
if [ "${1:-}" = "baselines" ]; then
    "${PYTHON}" scripts/local_structure/calculate_communities.py baselines "${SLURM_CPUS_PER_TASK}"
else
    "${PYTHON}" scripts/local_structure/calculate_communities.py "${SLURM_ARRAY_TASK_ID}" "${N_TASKS}" "${SLURM_CPUS_PER_TASK}"
fi
