"""
Compares each empirical network's GCC-fraction trajectory against its three
null-model counterparts (er, conf, sbm) at matching perturbation conditions.

GCC fraction is bounded in [0, 1], so unlike the Fiedler value (which is
strictly positive and can span orders of magnitude) a plain difference
(empirical - null) is the natural comparison here rather than a log ratio.

Reads directly from the per-network repeat-level files under
outputs/global_properties/gcc_singletons/gcc_singletons_per_network/, rather
than from the aggregated_gcc_singletons.csv produced by
aggregate_gcc_singleton_results.py, because that aggregation's regex-based
network-name extraction does not recognize the short baseline network_ids
(e.g. "astro_baseline") and silently collapses every network's baseline row
into a single bogus "unknown_network" bucket. Reading per-network files
avoids that bug since the network identity comes from the filename.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

output_filename = "outputs/global_properties/gcc_singletons/aggregated_gcc_diff_vs_null_models.csv"
per_network_dir = "outputs/global_properties/gcc_singletons/gcc_singletons_per_network"

root_networks = ["ppi", "astro", "power", "wiki"]
null_suffixes = ["er", "conf", "sbm"]

pattern = re.compile(r'_(add.*|remov.*)_noise_(\d+p\d+|\d+\.\d+|\d+)')


def load_conditions(net_key):
    path = os.path.join(per_network_dir, f"gcc_singletons_{net_key}.csv")
    df_raw = pd.read_csv(path)

    rows = []
    for perturbation_method, group in df_raw.groupby("perturbation_method"):
        if perturbation_method == "baseline":
            rows.append({
                "noise_type": "baseline",
                "action": "none",
                "noise_level": 0.0,
                "gcc_mean": group["gcc"].mean(),
                "gcc_std": 0.0,
            })
            continue

        extracted = group["network_id"].str.extract(pattern)
        extracted.columns = ["op_type", "noise_val"]
        group = pd.concat([group.reset_index(drop=True), extracted.reset_index(drop=True)], axis=1)

        unparsed = group[group["op_type"].isna()]
        for bad_id in unparsed["network_id"]:
            print(f"Could not parse noise level from network_id: {bad_id}")
        group = group.dropna(subset=["op_type", "noise_val"])

        group["action"] = np.where(group["op_type"].str.startswith("remo"), "removal", "addition")
        group["noise_level"] = group["noise_val"].str.replace("p", ".", regex=False).astype(float)

        for (action, noise_level), sub in group.groupby(["action", "noise_level"]):
            rows.append({
                "noise_type": perturbation_method,
                "action": action,
                "noise_level": noise_level,
                "gcc_mean": sub["gcc"].mean(),
                "gcc_std": sub["gcc"].std(),
            })

    return pd.DataFrame(rows)


all_network_ids = root_networks + [f"{root}_{suf}" for root in root_networks for suf in null_suffixes]
conditions = {net_key: load_conditions(net_key) for net_key in all_network_ids}

results = []
for root in root_networks:
    df_emp = conditions[root]

    for suf in null_suffixes:
        null_id = f"{root}_{suf}"
        df_null = conditions[null_id]

        merged = df_emp.merge(
            df_null,
            on=["noise_type", "action", "noise_level"],
            suffixes=("_empirical", "_null"),
            how="inner",
        )

        merged["network_id"] = root
        merged["null_model"] = suf
        merged["diff_gcc_vs_null"] = merged["gcc_mean_empirical"] - merged["gcc_mean_null"]
        merged["std_diff_gcc_vs_null"] = np.sqrt(
            merged["gcc_std_empirical"] ** 2 + merged["gcc_std_null"] ** 2
        )
        results.extend(merged.to_dict("records"))

df_final = pd.DataFrame(results)
cols = [
    "network_id", "null_model", "noise_type", "action", "noise_level",
    "gcc_mean_empirical", "gcc_mean_null",
    "diff_gcc_vs_null", "std_diff_gcc_vs_null",
]
df_final = df_final[cols]
df_final.sort_values(by=["network_id", "null_model", "noise_type", "noise_level"], inplace=True)
df_final.reset_index(drop=True, inplace=True)

df_final.to_csv(output_filename, index=False)

print(f"\nNull-model normalization complete! Saved to {output_filename}")
print("\nPreview of the normalized DataFrame:")
print(df_final.head(15).to_string())
