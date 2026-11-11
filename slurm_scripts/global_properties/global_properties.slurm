#!/bin/bash
#SBATCH --job-name=global
#SBATCH --array=0-59%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_80
#SBATCH --output=outputs/logs/%A_%a.out
#SBATCH --error=outputs/logs/err_%A_%a.txt

# Global properties of all perturbed networks, one array task per unit (the
# same 60 units as generating_perturbed_networks.slurm):
#   uv run python scripts/perturbed_networks/generating_perturbed_networks.py --list
#
# Resources: the largest units are the astro null models, 8,100 networks each.
# All metrics take about 1.7 s for the largest astro network locally (and the
# cluster nodes were about 3x slower), so at most 8,100 x 5 s / 4 CPUs = 3 h.
#
# Finished files are skipped, so a task that timed out can just be resubmitted:
#   sbatch --array=9 slurm_scripts/global_properties/global_properties.slurm
#
# Submit from the repository root:
#   mkdir -p outputs/logs && sbatch slurm_scripts/global_properties/global_properties.slurm
# Afterwards: uv run python scripts/global_properties/aggregate_global_properties.py

set -euo pipefail

# Python environment on the node-local disk (see generating_perturbed_networks.slurm)
ENV_DIR="${TMPDIR:-/tmp}/noise_env_${SLURM_JOB_ID}"
trap 'rm -rf "${ENV_DIR}"' EXIT
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
export UV_LINK_MODE=copy
uv sync --frozen --no-dev --quiet
PYTHON="${ENV_DIR}/bin/python"

# every worker is one process; no extra threads from numpy or igraph
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

UNITS=()
while IFS= read -r unit; do UNITS+=("${unit}"); done \
    < <("${PYTHON}" scripts/perturbed_networks/generating_perturbed_networks.py --list)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#UNITS[@]}" ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} out of range: only ${#UNITS[@]} units" >&2
    exit 1
fi
UNIT=${UNITS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID}: ${UNIT} on $(hostname)"

"${PYTHON}" scripts/global_properties/calculate_global_properties.py "${UNIT}" "${SLURM_CPUS_PER_TASK}"
