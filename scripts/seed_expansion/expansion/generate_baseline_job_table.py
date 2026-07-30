"""
Scans data/baseline_networks/*.csv and writes a job table, one row per
(network, algorithm), for the baseline SLURM array.

Usage:
    python notebooks/generate_baseline_job_table.py
"""
from pathlib import Path
import pandas as pd

BASELINE_DIR_1 = Path("data/baseline_networks")
BASELINE_DIR_2 = Path("data/baseline_networks/null_models")
ALGORITHMS = ["rwr_row", "rwr_sym", "diamond", "first_neighbors"]
OUT_PATH = Path("slurm_scripts/seed_expansion/expansion/jobs/baseline_jobs.csv")


def main():
    networks = sorted(p.stem for p in BASELINE_DIR_1.glob("*.csv")) + sorted(p.stem for p in BASELINE_DIR_2.glob("*.csv"))
    if not networks:
        raise SystemExit(f"No .csv files found in {BASELINE_DIR_1} or {BASELINE_DIR_2}")

    rows = []
    for net in networks:
        for algo in ALGORITHMS:
            rows.append({
                "network": net,
                "edgelist_path": str(BASELINE_DIR_1 / f"{net}.csv") if net in [p.stem for p in BASELINE_DIR_1.glob("*.csv")] else str(BASELINE_DIR_2 / f"{net}.csv"),
                "algorithm": algo,
            })

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} jobs ({len(networks)} networks x {len(ALGORITHMS)} algorithms) "
          f"to {OUT_PATH}")
    print("SLURM array should be: --array=1-%d" % len(df))


if __name__ == "__main__":
    main()