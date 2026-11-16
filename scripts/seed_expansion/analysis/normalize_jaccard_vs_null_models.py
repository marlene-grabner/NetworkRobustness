"""
Compares each empirical network's seed-expansion Jaccard-at-k=50 trajectory
against its three null-model counterparts (er, conf, sbm) at matching
perturbation conditions:
    diff(noise_level) = empirical_jaccard(noise_level) - null_jaccard(noise_level)

Jaccard is bounded in [0, 1], so a plain difference is the natural
comparison (like GCC fraction, global efficiency and ARI; unlike the
Fiedler value, which needs a log2 ratio because it is strictly positive and
can span orders of magnitude).

Run scripts/seed_expansion/analysis/aggregated_csvs.py first to produce the
input CSV this reads.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("outputs/seed_expansion/expansion/perturbed_summary_csvs/local_results_aggregated.csv")
OUTPUT_PATH = Path("outputs/seed_expansion/expansion/perturbed_summary_csvs/jaccard_k50_diff_vs_null_models.csv")

K_CATEGORY = "50"
root_networks = ["ppi", "astro", "power", "wiki"]
null_suffixes = ["er", "conf", "sbm"]


def main():
    if not INPUT_PATH.exists():
        sys.exit(f"{INPUT_PATH} not found. Run aggregated_csvs.py first.")

    df = pd.read_csv(INPUT_PATH)
    df = df[df["k_category"].astype(str) == K_CATEGORY]

    algorithms = sorted(df["algorithm"].unique())
    results = []

    for alg in algorithms:
        df_alg = df[df["algorithm"] == alg]

        for root in root_networks:
            df_emp = df_alg[df_alg["network"] == root]
            if df_emp.empty:
                print(f"Warning: no {alg}/k={K_CATEGORY} data for {root}, skipping.")
                continue

            for suf in null_suffixes:
                null_id = f"{root}_{suf}"
                df_null = df_alg[df_alg["network"] == null_id]
                if df_null.empty:
                    print(f"Warning: no {alg}/k={K_CATEGORY} data for {null_id}, skipping.")
                    continue

                merged = df_emp.merge(
                    df_null,
                    on=["perturbation_type", "modification_type", "noise_level"],
                    suffixes=("_empirical", "_null"),
                    how="inner",
                )

                merged["algorithm"] = alg
                merged["network_id"] = root
                merged["null_model"] = suf
                merged["diff_jaccard_k50_vs_null"] = (
                    merged["jaccard_mean_empirical"] - merged["jaccard_mean_null"]
                )
                merged["std_diff_jaccard_k50_vs_null"] = np.sqrt(
                    merged["jaccard_std_empirical"] ** 2 + merged["jaccard_std_null"] ** 2
                )
                results.extend(merged.to_dict("records"))

    if not results:
        sys.exit("No matching empirical/null-model conditions found; nothing to write.")

    df_final = pd.DataFrame(results)
    cols = [
        "algorithm", "network_id", "null_model", "perturbation_type", "modification_type", "noise_level",
        "jaccard_mean_empirical", "jaccard_mean_null",
        "diff_jaccard_k50_vs_null", "std_diff_jaccard_k50_vs_null",
    ]
    df_final = df_final[cols]
    df_final.sort_values(
        by=["algorithm", "network_id", "null_model", "perturbation_type", "noise_level"], inplace=True
    )
    df_final.reset_index(drop=True, inplace=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH, index=False)

    print(f"\nNull-model normalization complete! Saved to {OUTPUT_PATH}")
    print("\nPreview of the normalized DataFrame:")
    print(df_final.head(15).to_string())


if __name__ == "__main__":
    main()
