# submit_all.py  —  run once to submit all jobs
import os, subprocess

baseline_tsvs = {
    "ppi": "data/baseline_networks/chloe_ppi_lcc_2026_02_23.csv",
    "astro": "data/baseline_networks/ca-AstroPh_gcc.csv",
    "power":"data/baseline_networks/western_us_power_grid.csv",
    "wiki": "data/baseline_networks/wiki-Vote_gcc.csv",
    "ppi_er": "data/baseline_networks/null_models/chloe_ppi_erdos_renyi.csv",
    "ppi_conf": "data/baseline_networks/null_models/chloe_ppi_configuration_model.csv",
    "ppi_sbm": "data/baseline_networks/null_models/chloe_ppi_sbm.csv",
    "astro_er": "data/baseline_networks/null_models/ca-AstroPh_erdos_renyi.csv",
    "astro_conf": "data/baseline_networks/null_models/ca-AstroPh_configuration_model.csv",
    "astro_sbm": "data/baseline_networks/null_models/ca-AstroPh_sbm.csv",
    "power_er": "data/baseline_networks/null_models/western_us_power_grid_erdos_renyi.csv",
    "power_conf": "data/baseline_networks/null_models/western_us_power_grid_configuration_model.csv",
    "power_sbm": "data/baseline_networks/null_models/western_us_power_grid_sbm.csv",
    "wiki_er": "data/baseline_networks/null_models/wiki-Vote_erdos_renyi.csv",
    "wiki_conf": "data/baseline_networks/null_models/wiki-Vote_configuration_model.csv",
    "wiki_sbm": "data/baseline_networks/null_models/wiki-Vote_sbm.csv"
}

perturbed_folders = {
    "ppi": "data/perturbed_networks/chloe_ppi_lcc_2026_02_23",
    "astro": "data/perturbed_networks/ca-AstroPh_gcc",
    "power":"data/perturbed_networks/western_us_power_grid",
    "wiki": "data/perturbed_networks/wiki-Vote_gcc",
    "ppi_er": "data/perturbed_networks/chloe_ppi_lcc_2026_02_23_erdos_renyi",
    "ppi_conf": "data/perturbed_networks/chloe_ppi_lcc_2026_02_23_configuration_model",
    "ppi_sbm": "data/perturbed_networks/chloe_ppi_lcc_2026_02_23_sbm",
    "astro_er": "data/perturbed_networks/ca-AstroPh_erdos_renyi",
    "astro_conf": "data/perturbed_networks/ca-AstroPh_configuration_model",
    "astro_sbm": "data/perturbed_networks/ca-AstroPh_sbm",
    "power_er": "data/perturbed_networks/western_us_power_grid_erdos_renyi",
    "power_conf": "data/perturbed_networks/western_us_power_grid_configuration_model",
    "power_sbm": "data/perturbed_networks/western_us_power_grid_sbm",
    "wiki_er": "data/perturbed_networks/wiki-Vote_erdos_renyi",
    "wiki_conf": "data/perturbed_networks/wiki-Vote_configuration_model",
    "wiki_sbm": "data/perturbed_networks/wiki-Vote_sbm"
}

#baseline_tsvs = {"test": "data/baseline_networks/test_network/karate_club.csv"}
#perturbed_folders = {"test": "data/perturbed_networks/test_network"}

OUT_ROOT = './outputs/global_properties/algebraic_connectivity'
WORKER = './notebooks_general_analysis/global_properties/fiedler_value.py'

jobs = []  # collect (parquet, baseline, out_csv) triples

for net_key, baseline_tsv in baseline_tsvs.items():
    perturbed_root = perturbed_folders[net_key]
    for noise_type in os.listdir(perturbed_root):
        noise_dir = os.path.join(perturbed_root, noise_type)
        for fname in os.listdir(noise_dir):
            if not fname.endswith('.parquet'):
                continue
            parquet = os.path.join(noise_dir, fname)
            out_csv = os.path.join(OUT_ROOT, net_key, noise_type,
                                   fname.replace('.parquet', '.csv'))
            # Do not overwrite existing results
            if os.path.exists(out_csv):
                raise ValueError(f"Output already exists: {out_csv}")
            jobs.append((parquet, baseline_tsv, out_csv))



# Write a task file: one line per job
task_file = 'slurm_scripts/global_properties/tmp/fiedler_tasks.txt'
with open(task_file, 'w') as f:
    for parquet, baseline, out in jobs:
        f.write(f"{parquet}\t{baseline}\t{out}\n")

print(f"Total jobs to submit: {len(jobs)}")

# SLURM array job — one task per line in the file
slurm_script = f"""#!/bin/bash
#SBATCH --job-name=fiedler
#SBATCH --array=1-{len(jobs)}%10        # %10 = max 10 running at once, tune this
#SBATCH --cpus-per-task=1               # one core per job — SLURM IS your parallelism
#SBATCH --mem=8G                        # 20k nodes, sparse matrices — 8G is safe
#SBATCH --time=00:10:00                 # tune after benchmarking a few jobs
#SBATCH --output=outputs/logs/fiedler_%A_%a.out
#SBATCH --error=outputs/logs/fiedler_%A_%a.err

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file})
PARQUET=$(echo "$LINE" | cut -f1)
BASELINE=$(echo "$LINE" | cut -f2)
OUT=$(echo "$LINE" | cut -f3)

uv run {WORKER} "$PARQUET" "$BASELINE" "$OUT"
"""

script_path = 'slurm_scripts/global_properties/tmp/submit_fiedler.sh'
with open(script_path, 'w') as f:
    f.write(slurm_script)

subprocess.run(['sbatch', script_path])
print("Submitted.")
