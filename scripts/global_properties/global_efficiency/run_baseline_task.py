"""
One SLURM array task = one row of
slurm_scripts/global_properties/global_efficiency/jobs/baseline_jobs.csv
= one baseline network (or null model).

Usage (called by slurm_scripts/global_properties/global_efficiency/baseline_array.slurm):
    uv run scripts/global_properties/global_efficiency/run_baseline_task.py --job-row 3
"""
import argparse
import time
from pathlib import Path
import pandas as pd
from NoiseEffect.GlobalProperties import load_baseline_node_index, global_efficiency_from_edges

JOB_TABLE = Path("slurm_scripts/global_properties/global_efficiency/jobs/baseline_jobs.csv")
OUT_DIR = Path("outputs/global_properties/global_efficiency/baseline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-table", type=str, default=str(JOB_TABLE))
    ap.add_argument("--job-row", type=int, required=True)
    args = ap.parse_args()

    jobs = pd.read_csv(args.job_table)
    job = jobs.iloc[args.job_row - 1]
    network, edgelist_path = job["network"], job["edgelist_path"]

    t0 = time.time()
    print(f"Processing baseline network: {network} ({edgelist_path})")

    node_to_idx, n_nodes = load_baseline_node_index(edgelist_path)
    df_base = pd.read_csv(edgelist_path, sep=',', header=None, names=['source', 'target'], dtype=str)
    efficiency = global_efficiency_from_edges(df_base, node_to_idx, n_nodes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{network}.csv"
    pd.DataFrame([{
        "network": network,
        "n_nodes": n_nodes,
        "global_efficiency": efficiency,
    }]).to_csv(out_path, index=False)

    print(f"[{network}] n_nodes={n_nodes} global_efficiency={efficiency:.6g} "
          f"-> {out_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
