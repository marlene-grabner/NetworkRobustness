#!/bin/bash
# Submits the whole seed expansion analysis, each step waiting for the last:
#   seeds -> baselines -> perturbed array -> aggregate
#
# Run from the repository root, after the perturbed networks exist
# (data/perturbed_networks_ensembles/):
#   bash slurm_scripts/seed_expansion/submit_seed_expansion.sh
#
# Options:
#   --skip-seeds  keep the existing outputs/seed_expansion/synthetic_seeds/seed_sets.csv
#                 (new seeds invalidate all results in outputs/seed_expansion/results/)
#
# To resubmit only the array after a timeout (finished files are skipped):
#   sbatch slurm_scripts/seed_expansion/seed_expansion.slurm

set -euo pipefail
SCRIPT=slurm_scripts/seed_expansion/seed_expansion.slurm
mkdir -p outputs/logs

if [ "${1:-}" = "--skip-seeds" ]; then
    [ -f outputs/seed_expansion/synthetic_seeds/seed_sets.csv ] || { echo "no seed_sets.csv to keep" >&2; exit 1; }
    dependency=""
else
    if [ -d outputs/seed_expansion/results ]; then
        echo "outputs/seed_expansion/results/ exists and was computed with the old seed sets." >&2
        echo "Move or delete it first, or use --skip-seeds." >&2
        exit 1
    fi
    jid=$(sbatch --parsable --array=0 --time=00:30:00 "${SCRIPT}" seeds)
    echo "seeds:     ${jid}"
    dependency="--dependency=afterok:${jid}"
fi

jid=$(sbatch --parsable --array=0 --time=02:00:00 ${dependency} "${SCRIPT}" baselines)
echo "baselines: ${jid}"
jid=$(sbatch --parsable --dependency=afterok:${jid} "${SCRIPT}")
echo "array:     ${jid}"
# afterany: aggregate what finished even if some tasks timed out
jid=$(sbatch --parsable --array=0 --time=01:00:00 --mem=32G --dependency=afterany:${jid} "${SCRIPT}" aggregate)
echo "aggregate: ${jid}"
