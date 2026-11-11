"""
Combines the results of calculate_communities.py:

  outputs/local_structure/communities.parquet
    one row per perturbed network and algorithm
  outputs/local_structure/communities_baselines.csv
    one row per unperturbed network and algorithm
  outputs/local_structure/collapse_rates.csv
    per network, model, algorithm, noise type and level: the fraction of all
    runs (networks x seeds) that collapsed to one community, nearly collapsed
    (largest community >= 95% of the nodes) or left every node alone

Run from the repository root:
  uv run python scripts/local_structure/aggregate_communities.py
"""

import glob

import pandas as pd

RESULTS_DIR = "./outputs/local_structure/communities/"
OUT_DIR = "./outputs/local_structure/"
COUNTS = ["n_collapsed", "n_near_collapsed", "n_all_singletons"]

if __name__ == "__main__":
    files = sorted(glob.glob(RESULTS_DIR + "*/*/*.parquet"))
    perturbed = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    perturbed.to_parquet(OUT_DIR + "communities.parquet", index=False)

    files = sorted(glob.glob(RESULTS_DIR + "*/*/baseline.csv"))
    baselines = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    baselines.to_csv(OUT_DIR + "communities_baselines.csv", index=False)

    groups = ["network", "model", "algorithm", "noise_type", "level"]
    sums = perturbed.groupby(groups)[["n_seeds", *COUNTS]].sum()
    rates = sums[COUNTS].div(sums["n_seeds"], axis=0)
    rates.columns = [c.removeprefix("n_") + "_rate" for c in COUNTS]
    rates.insert(0, "n_runs", sums["n_seeds"])
    rates.reset_index().to_csv(OUT_DIR + "collapse_rates.csv", index=False)

    print(f"{len(perturbed)} rows from {len(files)} units")
    print(rates[rates.iloc[:, 1:].sum(axis=1) > 0].to_string())
