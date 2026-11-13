#!/bin/bash
#SBATCH --job-name=seed_expansion
#SBATCH --array=0-199%30
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_80
#SBATCH --output=outputs/logs/%A_%a.out
#SBATCH --error=outputs/logs/err_%A_%a.txt

# Seed expansion on all perturbed networks, in four steps (first argument):
#   seeds      seed sets of all unperturbed networks (bfs + Leiden communities),
#              outputs/seed_expansion/synthetic_seeds/seed_sets.csv
#   baselines  seed expansion on all unperturbed networks
#   (none)     array task i of N_TASKS: every N_TASKS-th perturbed file
#   aggregate  one table of all results
# submit_seed_expansion.sh submits all four with the right dependencies.
#
# The 4,860 perturbed files are spread over the array tasks (task i takes
# every 200th file), so every task gets a similar mix of small and large
# networks.
#
# Resources (local timings; 25 seed sets x 4 algorithms per network): seeds
# ~1 min, baselines ~20 CPU min. Perturbed: DIAMOnD dominates, ~11 s per
# network for the largest networks locally, most are much smaller. Upper
# bound ~200k perturbed networks x 11 s = 630 CPU hours locally, ~1,900 on the
# cluster if it is 3x slower, i.e. about 2.5 h per task on 4 CPUs; 8 h leaves
# room for tasks with several dense (addition 200%) or astro files.
#
# Finished files are skipped, so tasks that timed out can just be resubmitted.
# N_TASKS must stay the same as the array size.

set -euo pipefail
N_TASKS=200

# Python environment on the node-local disk (see generating_perturbed_networks.slurm)
ENV_DIR="${TMPDIR:-/tmp}/noise_env_${SLURM_JOB_ID}"
trap 'rm -rf "${ENV_DIR}"' EXIT
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
export UV_LINK_MODE=copy
uv sync --frozen --no-dev --quiet
PYTHON="${ENV_DIR}/bin/python"

# every worker is one process; no extra threads from numpy/scipy/igraph
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Array task ${SLURM_ARRAY_TASK_ID} on $(hostname), step ${1:-perturbed}"
case "${1:-}" in
    seeds)
        "${PYTHON}" scripts/seed_expansion/synthetic_seed_generation/synthetic_seed_generation.py "${SLURM_CPUS_PER_TASK}" ;;
    baselines)
        "${PYTHON}" scripts/seed_expansion/expansion/calculate_seed_expansion.py baselines "${SLURM_CPUS_PER_TASK}" ;;
    aggregate)
        "${PYTHON}" scripts/seed_expansion/expansion/aggregate_seed_expansion.py ;;
    "")
        "${PYTHON}" scripts/seed_expansion/expansion/calculate_seed_expansion.py "${SLURM_ARRAY_TASK_ID}" "${N_TASKS}" "${SLURM_CPUS_PER_TASK}" ;;
    *)
        echo "unknown step: $1" >&2; exit 1 ;;
esac
