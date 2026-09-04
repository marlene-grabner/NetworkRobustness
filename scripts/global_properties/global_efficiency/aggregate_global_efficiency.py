"""
Run once after the baseline array and the perturbed array have both finished.
Concatenates every outputs/global_properties/global_efficiency/baseline/*.csv
and outputs/global_properties/global_efficiency/perturbed/*.csv into a single
tidy CSV: network, noise_type, action, noise_level, repeat, global_efficiency.

Usage:
    uv run scripts/global_properties/global_efficiency/aggregate_global_efficiency.py
"""
from pathlib import Path
import pandas as pd

BASELINE_DIR = Path("outputs/global_properties/global_efficiency/baseline")
PERTURBED_DIR = Path("outputs/global_properties/global_efficiency/perturbed")
OUT_PATH = Path("outputs/global_properties/global_efficiency/global_efficiency_baseline_and_perturbed.csv")


def main():
    dfs = []

    baseline_files = sorted(BASELINE_DIR.glob("*.csv"))
    if not baseline_files:
        print(f"Warning: no baseline results found in {BASELINE_DIR}")
    for f in baseline_files:
        df = pd.read_csv(f)[["network", "global_efficiency"]]
        df["noise_type"] = "baseline"
        df["action"] = "none"
        df["noise_level"] = 0.0
        df["repeat"] = pd.NA
        dfs.append(df)

    perturbed_files = sorted(PERTURBED_DIR.glob("*.csv"))
    if not perturbed_files:
        print(f"Warning: no perturbed results found in {PERTURBED_DIR}")
    for f in perturbed_files:
        dfs.append(pd.read_csv(f))

    if not dfs:
        raise SystemExit("No results found to aggregate.")

    full = pd.concat(dfs, ignore_index=True)
    full = full[["network", "noise_type", "action", "noise_level", "repeat", "global_efficiency"]]
    full.sort_values(by=["network", "noise_type", "action", "noise_level", "repeat"], inplace=True)
    full.reset_index(drop=True, inplace=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_PATH, index=False)

    print(f"Aggregated {len(baseline_files)} baseline + {len(perturbed_files)} perturbed files "
          f"-> {len(full)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
