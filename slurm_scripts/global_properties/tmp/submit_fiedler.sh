#!/bin/bash
#SBATCH --job-name=fiedler
#SBATCH --array=1-864%10        # %10 = max 10 running at once, tune this
#SBATCH --cpus-per-task=1               # one core per job — SLURM IS your parallelism
#SBATCH --mem=8G                        # 20k nodes, sparse matrices — 8G is safe
#SBATCH --time=00:10:00                 # tune after benchmarking a few jobs
#SBATCH --output=outputs/logs/fiedler_%A_%a.out
#SBATCH --error=outputs/logs/fiedler_%A_%a.err

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm_scripts/global_properties/tmp/fiedler_tasks.txt)
PARQUET=$(echo "$LINE" | cut -f1)
BASELINE=$(echo "$LINE" | cut -f2)
OUT=$(echo "$LINE" | cut -f3)

uv run ./notebooks_general_analysis/global_properties/fiedler_value.py "$PARQUET" "$BASELINE" "$OUT"
