"""
Compares each empirical network's Fiedler-value trajectory against its three
null-model counterparts (er, conf, sbm) at matching perturbation conditions.

Since Fiedler values span orders of magnitude and are strictly positive, a
plain difference (empirical - null) would not treat "empirical is 2x the
null" and "empirical is half the null" symmetrically. So, like the
own-baseline normalization in normalize_algebraic_connectivity.py, this uses
log2(empirical / null) at each matching (noise_type, edge_operation,
noise_level) point.
"""
import os
import glob
import re
import numpy as np
import pandas as pd

output_filename = "outputs/global_properties/algebraic_connectivity/aggregated_fiedler_log2_ratio_vs_null_models.csv"

baseline_dir = "outputs/global_properties/algebraic_connectivity/baseline/baseline_algebraic_connectivity.csv"
pertubr_dir = "outputs/global_properties/algebraic_connectivity/perturbed"
perturbation_targets = ["perturbed_random_target", "perturbed_hub_target", "perturbed_periphery_target"]

root_networks = ["ppi", "astro", "power", "wiki"]
null_suffixes = ["er", "conf", "sbm"]

pattern = re.compile(r'_(add.*|remov.*)_noise_(\d+p\d+|\d+\.\d+|\d+)')


def load_conditions(net_id, baseline_val):
    """Per-(noise_type, edge_operation, noise_level) mean/std for one network."""
    rows = [{
        "noise_type": "baseline",
        "edge_operation": "none",
        "noise_level": 100.0,
        "avg_algebraic_connectivity": baseline_val,
        "std_algebraic_connectivity": 0.0,
    }]

    for target in perturbation_targets:
        folder_path = os.path.join(pertubr_dir, net_id, target)
        if not os.path.isdir(folder_path):
            continue

        for csv_file in glob.glob(os.path.join(folder_path, "*.csv")):
            filename = os.path.basename(csv_file)
            match = pattern.search(filename)
            if not match:
                print(f"Could not parse noise level from filename: {filename}")
                continue

            op_type = match.group(1)
            noise_val = float(match.group(2).replace('p', '.'))
            actual_noise_level = (100 - (noise_val * 100)) if op_type.startswith("remo") else (100 + noise_val * 100)

            df_pert = pd.read_csv(csv_file)
            rows.append({
                "noise_type": target,
                "edge_operation": op_type,
                "noise_level": actual_noise_level,
                "avg_algebraic_connectivity": df_pert['algebraic_connectivity'].mean(),
                "std_algebraic_connectivity": df_pert['algebraic_connectivity'].std(),
            })

    return pd.DataFrame(rows)


df_base = pd.read_csv(baseline_dir)
baseline_lookup = dict(zip(df_base["network_id"], df_base["algebraic_connectivity"]))

all_network_ids = root_networks + [f"{root}_{suf}" for root in root_networks for suf in null_suffixes]
conditions = {}
for net_id in all_network_ids:
    baseline_val = baseline_lookup.get(net_id)
    if baseline_val is None:
        print(f"Warning: no baseline value for {net_id}, skipping.")
        continue
    conditions[net_id] = load_conditions(net_id, baseline_val)

results = []
for root in root_networks:
    if root not in conditions:
        continue
    df_emp = conditions[root]

    for suf in null_suffixes:
        null_id = f"{root}_{suf}"
        if null_id not in conditions:
            print(f"Warning: no data for null model {null_id}, skipping.")
            continue
        df_null = conditions[null_id]

        merged = df_emp.merge(
            df_null,
            on=["noise_type", "edge_operation", "noise_level"],
            suffixes=("_empirical", "_null"),
            how="inner",
        )

        log2_ratio = np.log2(
            merged["avg_algebraic_connectivity_empirical"] / merged["avg_algebraic_connectivity_null"]
        )

        # Error propagation for log2(A/B) from each side's repeat-to-repeat std
        # (relative errors added in quadrature, converted from ln to log2).
        rel_err_sq = (
            (merged["std_algebraic_connectivity_empirical"] / merged["avg_algebraic_connectivity_empirical"]) ** 2
            + (merged["std_algebraic_connectivity_null"] / merged["avg_algebraic_connectivity_null"]) ** 2
        )
        log2_ratio_std = np.sqrt(rel_err_sq) / np.log(2)

        merged["network_id"] = root
        merged["null_model"] = suf
        merged["log2_ratio_vs_null"] = log2_ratio
        merged["std_log2_ratio_vs_null"] = log2_ratio_std
        results.extend(merged.to_dict("records"))

df_final = pd.DataFrame(results)
cols = [
    "network_id", "null_model", "noise_type", "edge_operation", "noise_level",
    "avg_algebraic_connectivity_empirical", "avg_algebraic_connectivity_null",
    "log2_ratio_vs_null", "std_log2_ratio_vs_null",
]
df_final = df_final[cols]
df_final.sort_values(by=["network_id", "null_model", "noise_type", "noise_level"], inplace=True)
df_final.reset_index(drop=True, inplace=True)

df_final.to_csv(output_filename, index=False)

print(f"\nNull-model normalization complete! Saved to {output_filename}")
print("\nPreview of the normalized DataFrame:")
print(df_final.head(15).to_string())
