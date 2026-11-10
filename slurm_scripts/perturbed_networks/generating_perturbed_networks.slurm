#!/bin/bash
#SBATCH --job-name=noise
#SBATCH --array=0-59%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_80
#SBATCH --output=outputs/logs/%A_%a.out
#SBATCH --error=outputs/logs/err_%A_%a.txt

# One array task per unit: an empirical network (4 core + 44 extended set) or
# one null model of a core network (4 x er/conf/sbm, 10 replicates each).
# The unit list comes from the script itself, so the order is always the same:
#   uv run python scripts/perturbed_networks/generating_perturbed_networks.py --list
# 60 units -> --array=0-59. Rerun single units with e.g. sbatch --array=5,17 ...
#
# Resources, measured locally (single-threaded code, so 1 CPU):
#   largest unit = astro: empirical 18 min, 3.3 GB peak. A null model unit has
#   4x as many networks (10 replicates x 10 repeats vs 25 repeats), so about
#   70 min locally; its peak memory is lower (10 repeats held at once, not 25).
#   4 h / 6 GB leaves room for a slower node. All other units are smaller.
#
# Submit from the repository root (warm the uv cache once on the login node
# with `uv sync`, so the array tasks only copy packages instead of downloading):
#   mkdir -p outputs/logs && sbatch slurm_scripts/perturbed_networks/generating_perturbed_networks.slurm

set -euo pipefail

# Python environment on the node-local disk: many small file reads (imports)
# are slow on the shared filesystem. Packages come from the shared uv cache.
ENV_DIR="${TMPDIR:-/tmp}/noise_env_${SLURM_JOB_ID}"
trap 'rm -rf "${ENV_DIR}"' EXIT
export UV_PROJECT_ENVIRONMENT="${ENV_DIR}"
export UV_LINK_MODE=copy  # cache and environment are on different filesystems
uv sync --frozen --no-dev --quiet
PYTHON="${ENV_DIR}/bin/python"

UNITS=()
while IFS= read -r unit; do UNITS+=("${unit}"); done \
    < <("${PYTHON}" scripts/perturbed_networks/generating_perturbed_networks.py --list)

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#UNITS[@]}" ]; then
    echo "Array index ${SLURM_ARRAY_TASK_ID} out of range: only ${#UNITS[@]} units" >&2
    exit 1
fi
UNIT=${UNITS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID}: ${UNIT} on $(hostname)"

"${PYTHON}" scripts/perturbed_networks/generating_perturbed_networks.py "${UNIT}"
