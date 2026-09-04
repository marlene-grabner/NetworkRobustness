"""
One SLURM array task = one row of
slurm_scripts/global_properties/global_efficiency/jobs/perturbed_jobs.csv
= one (network, noise_type, action, noise_level) parquet file, which itself
holds all 100 repeats.

The node index (including isolated nodes pulled from the baseline's sidecar
file) is built once from the baseline edgelist and reused for every repeat,
since a perturbed edgelist can be missing nodes that ended up with no edges
in that particular repeat.

Global efficiency is a full all-pairs-shortest-paths style computation, so
it is the expensive step here (tens of seconds per repeat on the largest
networks, e.g. ppi/astro). The 100 repeats in a file are independent of each
other, so they are farmed out across SLURM_CPUS_PER_TASK worker processes.

All 100 repeat rows for the task are written to a single CSV, so a
per-file array produces one output file per row (not one per repeat). Run
aggregate_global_efficiency.py afterwards to combine everything, baseline
included, into one overview CSV.

Usage (called by slurm_scripts/global_properties/global_efficiency/perturbed_array.slurm):
    uv run scripts/global_properties/global_efficiency/run_perturbed_task.py --job-row 3
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from NoiseEffect.GlobalProperties import load_baseline_node_index, global_efficiency_from_edges

JOB_TABLE = Path("slurm_scripts/global_properties/global_efficiency/jobs/perturbed_jobs.csv")
OUT_DIR = Path("outputs/global_properties/global_efficiency/perturbed")


def _compute_repeat(repeat_id, group, node_to_idx, n_nodes, network, noise_type, action, noise_level):
    efficiency = global_efficiency_from_edges(group, node_to_idx, n_nodes)
    return {
        "network": network,
        "noise_type": noise_type,
        "action": action,
        "noise_level": noise_level,
        "repeat": int(repeat_id),
        "global_efficiency": efficiency,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-table", type=str, default=str(JOB_TABLE))
    ap.add_argument("--job-row", type=int, required=True)
    args = ap.parse_args()

    jobs = pd.read_csv(args.job_table)
    job = jobs.iloc[args.job_row - 1]
    network = job["network"]
    noise_type = job["noise_type"]
    action = job["action"]
    noise_level = job["noise_level"]
    edgelist_path = job["edgelist_path"]
    parquet_path = job["parquet_path"]

    t0 = time.time()

    node_to_idx, n_nodes = load_baseline_node_index(edgelist_path)
    df_pert = pd.read_parquet(parquet_path)

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    rows = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _compute_repeat, repeat_id, group, node_to_idx, n_nodes,
                network, noise_type, action, noise_level,
            ): repeat_id
            for repeat_id, group in df_pert.groupby("repeat")
        }
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda r: r["repeat"])

    noise_level_str = str(noise_level).replace(".", "p")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{network}_{noise_type}_{action}_{noise_level_str}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"[{network}/{noise_type}/{action}/noise={noise_level}] "
          f"{len(rows)} repeats ({num_workers} workers) -> {out_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
