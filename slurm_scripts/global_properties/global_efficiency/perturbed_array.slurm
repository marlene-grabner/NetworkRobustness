#!/bin/bash
#SBATCH --job-name=global_efficiency_perturbed
#SBATCH --output=outputs/logs/global_efficiency_perturbed_%A_%a.out
#SBATCH --error=outputs/logs/global_efficiency_perturbed_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --array=1-864%20
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,TIME_LIMIT,TIME_LIMIT_80

# Run scripts/global_properties/global_efficiency/generate_perturbed_job_table.py
# once beforehand (after the perturbed parquet files exist under
# data/perturbed_networks/) to (re)create
# slurm_scripts/global_properties/global_efficiency/jobs/perturbed_jobs.csv and
# print the correct --array range. --array=1-864 assumes the current 16
# networks x 3 perturbation targets x 18 (noise_type x noise_level)
# files each; update it to match what the job table generator prints.
#
# Global efficiency needs a full shortest-path sweep per repeat, so this is
# the expensive step (benchmarked locally at ~16s/repeat on the largest
# networks, ppi/astro, at baseline density -- noisier/denser repeats will be
# slower). Each of the 100 repeats in a file is independent, so
# run_perturbed_task.py farms them out across SLURM_CPUS_PER_TASK worker
# processes; tune --cpus-per-task/--mem/--time after benchmarking a few jobs
# for your actual noise levels.
#
# MUST run after the baseline array has completed if you want the final
# aggregate CSV to include baseline rows (this script itself does not need
# the baseline results, only the aggregation step does). Submit with a
# dependency if desired, e.g.:
#   BASE_JOBID=$(sbatch --parsable slurm_scripts/global_properties/global_efficiency/baseline_array.slurm)
#   sbatch --dependency=afterok:${BASE_JOBID} slurm_scripts/global_properties/global_efficiency/perturbed_array.slurm

set -euo pipefail

uv run scripts/global_properties/global_efficiency/run_perturbed_task.py --job-row "${SLURM_ARRAY_TASK_ID}"
