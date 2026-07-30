"""
Run once after all perturbed array tasks finish. Concatenates every
results/perturbed_metrics/*.parquet into a single tidy parquet for analysis.

Usage:
    python notebooks/aggregate_perturbed_metrics.py
"""
from pathlib import Path
import pandas as pd

IN_DIR = Path("results/perturbed_metrics")
OUT_PATH = Path("results/perturbed_metrics_all.parquet")


def main():
    files = sorted(IN_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No files found in {IN_DIR}")

    dfs = [pd.read_parquet(f) for f in files]
    full = pd.concat(dfs, ignore_index=True)
    full.to_parquet(OUT_PATH, index=False)

    print(f"Concatenated {len(files)} files -> {len(full)} rows -> {OUT_PATH}")
    print(full.groupby(["network", "perturbation_type"]).size().head(20))


if __name__ == "__main__":
    main()