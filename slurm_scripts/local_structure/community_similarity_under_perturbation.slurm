#!/bin/bash
#SBATCH --job-name=community_%a
#SBATCH --array=0-1          # 3 noise_types × 20 levels
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%A_%a.out

ALGO="leiden"       # pass as env var or hardcode; submit once per algo
GRAPH_ID=0          # submit once per baseline graph, or loop inside

uv run notebooks_general_analysis/local_structure/community_similarity_under_perturbation.py \
    --task-id $SLURM_ARRAY_TASK_ID \
    --algo $ALGO \
    --graph-id $GRAPH_ID \
    --parquet-dir /path/to/parquets \
    --baseline-dir /path/to/baselines \
    --output-dir /path/to/output \
    --n-jobs $SLURM_CPUS_PER_TASK