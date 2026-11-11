"""
Combines the results of calculate_global_properties.py into two tables:

  outputs/global_properties/global_properties.parquet
    one row per perturbed network
    columns: network, model, noise_type, target, action, level, replicate, repeat, metrics
  outputs/global_properties/global_properties_baselines.csv
    one row per unperturbed network, columns: network, model, replicate, metrics

Run from the repository root:
  uv run python scripts/global_properties/aggregate_global_properties.py
"""

import glob

import pandas as pd

RESULTS_DIR = "./outputs/global_properties/all_metrics/"
OUT_DIR = "./outputs/global_properties/"

if __name__ == "__main__":
    files = sorted(glob.glob(RESULTS_DIR + "*/*/*.parquet"))
    perturbed = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    perturbed.to_parquet(OUT_DIR + "global_properties.parquet", index=False)

    files = sorted(glob.glob(RESULTS_DIR + "*/*/baseline.csv"))
    baselines = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    baselines.to_csv(OUT_DIR + "global_properties_baselines.csv", index=False)

    print(f"{len(perturbed)} perturbed and {len(baselines)} unperturbed networks")
    print(perturbed.groupby(["network", "model"]).size().unstack().to_string())
