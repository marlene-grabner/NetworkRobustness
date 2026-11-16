"""
Compares each empirical network's global-efficiency trajectory against its
three null-model counterparts (er, conf, sbm) at matching perturbation
conditions: diff(noise_level) = empirical(noise_level) - null(noise_level).

Global efficiency is bounded in [0, 1], so a plain difference is the natural
comparison (unlike the Fiedler value, which needs a log2 ratio because it is
strictly positive and can span orders of magnitude).

Run aggregate_global_efficiency.py first to produce the input CSV.

Usage:
    uv run scripts/global_properties/global_efficiency/analysis/normalize_global_efficiency_vs_null_models.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("outputs/global_properties/global_efficiency/global_efficiency_baseline_and_perturbed.csv")
OUTPUT_PATH = Path("outputs/global_properties/global_efficiency/global_efficiency_diff_vs_null_models.csv")

root_networks = ["ppi", "astro", "power", "wiki"]
null_suffixes = ["er", "conf", "sbm"]


def main():
    if not INPUT_PATH.exists():
        sys.exit(
            f"{INPUT_PATH} not found. Run aggregate_global_efficiency.py first "
            "(after both the baseline and perturbed SLURM arrays have finished)."
        )

    df_raw = pd.read_csv(INPUT_PATH)

    conditions = (
        df_raw.groupby(["network", "noise_type", "action", "noise_level"])["global_efficiency"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    conditions["std"] = conditions["std"].fillna(0.0)  # baseline has a single value, no repeats

    results = []
    for root in root_networks:
        df_emp = conditions[conditions["network"] == root]
        if df_emp.empty:
            print(f"Warning: no global efficiency data for {root}, skipping.")
            continue

        for suf in null_suffixes:
            null_id = f"{root}_{suf}"
            df_null = conditions[conditions["network"] == null_id]
            if df_null.empty:
                print(f"Warning: no global efficiency data for {null_id}, skipping.")
                continue

            merged = df_emp.merge(
                df_null,
                on=["noise_type", "action", "noise_level"],
                suffixes=("_empirical", "_null"),
                how="inner",
            )

            merged["network_id"] = root
            merged["null_model"] = suf
            merged["diff_global_efficiency_vs_null"] = merged["mean_empirical"] - merged["mean_null"]
            merged["std_diff_global_efficiency_vs_null"] = np.sqrt(
                merged["std_empirical"] ** 2 + merged["std_null"] ** 2
            )
            results.extend(merged.to_dict("records"))

    if not results:
        sys.exit("No matching empirical/null-model conditions found; nothing to write.")

    df_final = pd.DataFrame(results)
    df_final = df_final.rename(columns={"mean_empirical": "avg_global_efficiency_empirical",
                                         "mean_null": "avg_global_efficiency_null"})
    cols = [
        "network_id", "null_model", "noise_type", "action", "noise_level",
        "avg_global_efficiency_empirical", "avg_global_efficiency_null",
        "diff_global_efficiency_vs_null", "std_diff_global_efficiency_vs_null",
    ]
    df_final = df_final[cols]
    df_final.sort_values(by=["network_id", "null_model", "noise_type", "noise_level"], inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)

    print(f"\nNull-model normalization complete! Saved to {OUTPUT_PATH}")
    print("\nPreview of the normalized DataFrame:")
    print(df_final.head(15).to_string())


if __name__ == "__main__":
    main()
