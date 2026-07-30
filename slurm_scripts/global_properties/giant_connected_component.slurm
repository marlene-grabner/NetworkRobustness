#!/bin/bash
#SBATCH --job-name=gcc_singletons
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=42G
#SBATCH --time=05:00:00
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_80
#SBATCH --output=outputs/logs/out_%j.out
#SBATCH --error=outputs/logs/err_%j.txt



uv run python notebooks_general_analysis/global_properties/giant_connected_component.py