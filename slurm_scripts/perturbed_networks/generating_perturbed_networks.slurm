#!/bin/bash
#SBATCH --job-name=noise
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_80
#SBATCH --output=outputs/logs/%A_%a.out
#SBATCH --error=outputs/logs/err_%A_%a.txt

NETWORKS=(
    "chloe_ppi_lcc_2026_02_23"
    "chloe_ppi_lcc_2026_02_23_er"
    "chloe_ppi_lcc_2026_02_23_config"
    "chloe_ppi_lcc_2026_02_23_sbm"
    "ca-AstroPH_gcc"
    "ca-AstroPH_gcc_er"
    "ca-AstroPH_gcc_config"
    "ca-AstroPH_gcc_sbm"
)

NETWORK=${NETWORKS[$SLURM_ARRAY_TASK_ID]}


uv run python notebooks_general_analysis/perturbed_networks/generating_perturbed_networks.py $NETWORK