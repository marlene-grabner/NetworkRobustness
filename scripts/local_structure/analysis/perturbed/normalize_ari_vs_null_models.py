"""
Compares each empirical network's community-stability trajectory (ARI of the
perturbed partition vs. that network's own baseline partition) against its
three null-model counterparts (er, conf, sbm) at matching perturbation
conditions: diff(level) = empirical_vs_baseline_ari(level) - null_vs_baseline_ari(level).

ARI is bounded in [-1, 1] (in practice mostly [0, 1] here), so a plain
difference is the natural comparison, like GCC fraction and global
efficiency (unlike the Fiedler value, which needs a log2 ratio).

Run scripts/local_structure/analysis/perturbed/csv_perturbed_files.py first
to produce the per-algorithm aggregated CSVs this reads.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR = Path("outputs/local_structure/overview_csvs/perturbed")
OUTPUT_PATH = Path("outputs/local_structure/overview_csvs/perturbed/ari_diff_vs_null_models.csv")

algorithms = ["infomap", "leiden", "louvain", "label_propagation"]
root_networks = ["ppi", "astro", "power", "wiki"]
null_suffixes = ["er", "conf", "sbm"]


def main():
    results = []

    for alg in algorithms:
        input_path = INPUT_DIR / f"{alg}_aggregated.csv"
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping {alg}.")
            continue

        df = pd.read_csv(input_path)

        for root in root_networks:
            df_emp = df[df["base_network_name"] == root]
            if df_emp.empty:
                print(f"Warning: no {alg} data for {root}, skipping.")
                continue

            for suf in null_suffixes:
                null_id = f"{root}_{suf}"
                df_null = df[df["base_network_name"] == null_id]
                if df_null.empty:
                    print(f"Warning: no {alg} data for {null_id}, skipping.")
                    continue

                merged = df_emp.merge(
                    df_null,
                    on=["target", "type_of_noise", "level"],
                    suffixes=("_empirical", "_null"),
                    how="inner",
                )

                merged["algorithm"] = alg
                merged["network_id"] = root
                merged["null_model"] = suf
                merged["diff_vs_baseline_ari_vs_null"] = (
                    merged["vs_baseline_ari_mean_empirical"] - merged["vs_baseline_ari_mean_null"]
                )
                merged["std_diff_vs_baseline_ari_vs_null"] = np.sqrt(
                    merged["vs_baseline_ari_std_empirical"] ** 2 + merged["vs_baseline_ari_std_null"] ** 2
                )
                results.extend(merged.to_dict("records"))

    if not results:
        sys.exit("No matching empirical/null-model conditions found; nothing to write.")

    df_final = pd.DataFrame(results)
    cols = [
        "algorithm", "network_id", "null_model", "target", "type_of_noise", "level",
        "vs_baseline_ari_mean_empirical", "vs_baseline_ari_mean_null",
        "diff_vs_baseline_ari_vs_null", "std_diff_vs_baseline_ari_vs_null",
    ]
    df_final = df_final[cols]
    df_final.sort_values(by=["algorithm", "network_id", "null_model", "target", "level"], inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    df_final.to_csv(OUTPUT_PATH, index=False)

    print(f"\nNull-model normalization complete! Saved to {OUTPUT_PATH}")
    print("\nPreview of the normalized DataFrame:")
    print(df_final.head(15).to_string())


if __name__ == "__main__":
    main()
