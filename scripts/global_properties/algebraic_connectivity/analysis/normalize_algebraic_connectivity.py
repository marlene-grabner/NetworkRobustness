import os
import glob
import re
import numpy as np
import pandas as pd

# ============================================================================
# --- 0. Define Paths and Network Keys ---
# ===========================================================================

# Specify where to save the normalized results
output_filename = "outputs/global_properties/algebraic_connectivity/aggregated_mean_fiedler_values_perturbed_plus_baseline_log2_normalized.csv"
############################

baseline_dir = "outputs/global_properties/algebraic_connectivity/baseline/baseline_algebraic_connectivity.csv"
pertubr_dir = "outputs/global_properties/algebraic_connectivity/perturbed"
perturbation_targets = ["perturbed_random_target", "perturbed_hub_target", "perturbed_periphery_target"]
network_keys = [
    "ppi", "astro", "power", "wiki",
    "ppi_er", "ppi_conf", "ppi_sbm",
    "astro_er", "astro_conf", "astro_sbm",
    "power_er", "power_conf", "power_sbm",
    "wiki_er", "wiki_conf", "wiki_sbm"
]

# Fiedler value is always taken on the GCC, so a plain ratio to baseline
# treats an increase and a decrease of the same magnitude asymmetrically
# (e.g. x2 vs x0.5). log2(value / baseline) makes the two symmetric around 0.
aggregated_data = []

# ============================================================================
# --- 1. Load Baseline Data ---
# ===========================================================================

df_base = pd.read_csv(baseline_dir)
baseline_lookup = dict(zip(df_base["network_id"], df_base["algebraic_connectivity"]))

for _, row in df_base.iterrows():
    aggregated_data.append({
        "network_id": row["network_id"],
        "noise_type": "baseline",
        "edge_operation": "none",
        "noise_level": 100,
        "baseline_algebraic_connectivity": row["algebraic_connectivity"],
        "avg_algebraic_connectivity": row["algebraic_connectivity"],
        "std_algebraic_connectivity": 0.0,
        "avg_log2_ratio_fiedler": 0.0,
        "std_log2_ratio_fiedler": 0.0,
    })

print("Successfully loaded baseline data.")

# ============================================================================
# --- 2. Load Perturbed Repeats and Compute the log2 Ratio to Baseline ---
# ===========================================================================

# Regex to extract addition/removal and the noise value (e.g., handles "noise_0p20" or "noise_0.20")
pattern = re.compile(r'_(add.*|remov.*)_noise_(\d+p\d+|\d+\.\d+|\d+)')

for net_id in network_keys:
    baseline_val = baseline_lookup.get(net_id)
    if baseline_val is None:
        print(f"Warning: no baseline value for {net_id}, skipping its perturbed data.")
        continue

    for target in perturbation_targets:
        folder_path = os.path.join(pertubr_dir, net_id, target)

        if not os.path.isdir(folder_path):
            continue

        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            match = pattern.search(filename)

            if not match:
                print(f"Could not parse noise level from filename: {filename}")
                continue

            op_type = match.group(1)
            noise_val = float(match.group(2).replace('p', '.'))
            actual_noise_level = (100 - (noise_val * 100)) if op_type.startswith("remo") else (100 + noise_val * 100)

            try:
                # Read the CSV containing the repeats for this noise level
                df_pert = pd.read_csv(csv_file)

                # log2 ratio per repeat, then aggregate over repeats
                log2_ratio = np.log2(df_pert['algebraic_connectivity'] / baseline_val)

                aggregated_data.append({
                    "network_id": net_id,
                    "noise_type": target,
                    "edge_operation": op_type,
                    "noise_level": actual_noise_level,
                    "baseline_algebraic_connectivity": baseline_val,
                    "avg_algebraic_connectivity": df_pert['algebraic_connectivity'].mean(),
                    "std_algebraic_connectivity": df_pert['algebraic_connectivity'].std(),
                    "avg_log2_ratio_fiedler": log2_ratio.mean(),
                    "std_log2_ratio_fiedler": log2_ratio.std(),
                })
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

# ============================================================================
# --- 3. Save the Normalized DataFrame to CSV ---
# ===========================================================================

df_final = pd.DataFrame(aggregated_data)

df_final.sort_values(by=["network_id", "noise_type", "noise_level"], inplace=True)
df_final.reset_index(drop=True, inplace=True)

df_final.to_csv(output_filename, index=False)

print(f"\nNormalization complete! Saved to {output_filename}")
print("\nPreview of the normalized DataFrame:")
print(df_final.head(15).to_string())
