"""
Scans data/baseline_networks/*.csv and data/baseline_networks/null_models/*.csv
and writes a job table, one row per network, for the baseline SLURM array.

Usage:
    uv run scripts/global_properties/global_efficiency/generate_baseline_job_table.py
"""
from pathlib import Path
import pandas as pd

BASELINE_DIR = Path("data/baseline_networks")
NULL_MODEL_DIR = Path("data/baseline_networks/null_models")
OUT_PATH = Path("slurm_scripts/global_properties/global_efficiency/jobs/baseline_jobs.csv")


def main():
    rows = []
    for edgelist_path in sorted(BASELINE_DIR.glob("*.csv")) + sorted(NULL_MODEL_DIR.glob("*.csv")):
        if edgelist_path.name.endswith("_isolated_nodes.csv"):
            continue
        rows.append({
            "network": edgelist_path.stem,
            "edgelist_path": str(edgelist_path),
        })

    if not rows:
        raise SystemExit(f"No .csv files found in {BASELINE_DIR} or {NULL_MODEL_DIR}")

    df = pd.DataFrame(rows).sort_values("network")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} jobs to {OUT_PATH}")
    print("SLURM array should be: --array=1-%d" % len(df))


if __name__ == "__main__":
    main()
