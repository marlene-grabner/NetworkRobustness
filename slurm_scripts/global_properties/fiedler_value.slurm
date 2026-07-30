#!/bin/bash
#SBATCH --job-name=fiedler_value
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_80
#SBATCH --output=outputs/logs/out_%j.out
#SBATCH --error=outputs/logs/err_%j.txt

export PYTHONUNBUFFERED=TRUE


uv run python notebooks_general_analysis/global_properties/fiedler_value.py
