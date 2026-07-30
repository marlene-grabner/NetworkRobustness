import os
import glob
import re
import pandas as pd

# ============================================================================
# --- 0. Define Paths and Network Keys ---
# ===========================================================================

# Specify where to save final results
output_filename = "outputs/global_properties/algebraic_connectivity/aggregated_mean_fiedler_values_perturbed_plus_baseline.csv"
############################

baseline_dir = "outputs/global_properties/algebraic_connectivity/baseline/baseline_algebraic_connectivity.csv" # Where your network_id folders are located
pertubr_dir = "outputs/global_properties/algebraic_connectivity/perturbed"
perturbation_targets = ["perturbed_random_target", "perturbed_hub_target", "perturbed_periphery_target"]
network_keys = [
    "ppi", "astro", "power", "wiki", 
    "ppi_er", "ppi_conf", "ppi_sbm", 
    "astro_er", "astro_conf", "astro_sbm", 
    "power_er", "power_conf", "power_sbm", 
    "wiki_er", "wiki_conf", "wiki_sbm"
]

aggregated_data = []

# ============================================================================
# --- 1. Load and Format Baseline Data ---
# ===========================================================================

try:
    df_base = pd.read_csv(baseline_dir)
    for _, row in df_base.iterrows():
        aggregated_data.append({
            "network_id": row["network_id"],
            "noise_type": "baseline",
            "edge_operation": "none",
            "noise_level": 100,
            "avg_algebraic_connectivity": row["algebraic_connectivity"],
            "std_algebraic_connectivity": 0.0  # No standard deviation for a single baseline graph
        })
    print("Successfully loaded baseline data.")
except FileNotFoundError:
    print("Warning: baseline_algebraic_connectivity.csv not found in the current directory.")


# ============================================================================
# --- 2. Load, Calculate, and Format Perturbed Data ---
# ===========================================================================

# Regex to extract addition/removal and the noise value (e.g., handles "noise_0p20" or "noise_0.20")
pattern = re.compile(r'_(add.*|remov.*)_noise_(\d+p\d+|\d+\.\d+|\d+)')

for net_id in network_keys:
    for target in perturbation_targets:
        folder_path = os.path.join(pertubr_dir, net_id, target)
        
        # Skip if the directory doesn't exist
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
            # Convert string like "0p2" to float 0.2
            noise_val = float(match.group(2).replace('p', '.'))
            
            # Map removal to negative values, addition to positive values
            actual_noise_level = (100 - (noise_val*100)) if op_type.startswith("remo") else (100 + noise_val*100)
            
            try:
                # Read the CSV containing your 100 repeats
                df_pert = pd.read_csv(csv_file)
                
                # Calculate mean and std over the repeats
                avg_conn = df_pert['algebraic_connectivity'].mean()
                std_conn = df_pert['algebraic_connectivity'].std()
                
                aggregated_data.append({
                    "network_id": net_id,
                    "noise_type": target,
                    "edge_operation": op_type,
                    "noise_level": actual_noise_level,
                    "avg_algebraic_connectivity": avg_conn,
                    "std_algebraic_connectivity": std_conn
                })
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")


# ============================================================================
# --- 3. Save the Aggregated DataFrame to CSV ---
# ===========================================================================

df_final = pd.DataFrame(aggregated_data)

# Sort the dataframe so it's neat and logical
df_final.sort_values(by=["network_id", "noise_type", "noise_level"], inplace=True)

# Reset index after sorting
df_final.reset_index(drop=True, inplace=True)

# Save the final aggregated data to a new CSV
df_final.to_csv(output_filename, index=False)

print(f"\nAggregation complete! Saved to {output_filename}")
print("\nPreview of the aggregated DataFrame:")
print(df_final.head(15).to_string())