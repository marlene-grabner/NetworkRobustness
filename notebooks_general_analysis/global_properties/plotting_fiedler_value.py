#%%
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
#%%
# Configuration
baseline_dir = "../../outputs/global_properties/algebraic_connectivity/baseline/baseline_algebraic_connectivity.csv" # Where your network_id folders are located
pertubr_dir = "../../outputs/global_properties/algebraic_connectivity/perturbed"
perturbation_targets = ["perturbed_random_target", "perturbed_hub_target", "perturbed_periphery_target"]
network_keys = [
    "ppi", "astro", "power", "wiki", 
    "ppi_er", "ppi_conf", "ppi_sbm", 
    "astro_er", "astro_conf", "astro_sbm", 
    "power_er", "power_conf", "power_sbm", 
    "wiki_er", "wiki_conf", "wiki_sbm"
]

aggregated_data = []

# --- 1. Load and Format Baseline Data ---
try:
    df_base = pd.read_csv(baseline_dir)
    for _, row in df_base.iterrows():
        aggregated_data.append({
            "network_id": row["network_id"],
            "noise_type": "baseline",
            "edge_operation": "none",
            "noise_level": 0.0,
            "avg_algebraic_connectivity": row["algebraic_connectivity"],
            "std_algebraic_connectivity": 0.0  # No standard deviation for a single baseline graph
        })
    print("Successfully loaded baseline data.")
except FileNotFoundError:
    print("Warning: baseline_algebraic_connectivity.csv not found in the current directory.")


# --- 2. Load, Calculate, and Format Perturbed Data ---
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

# --- 3. Finalize DataFrame ---
df_final = pd.DataFrame(aggregated_data)

# Sort the dataframe so it's neat and logical
df_final.sort_values(by=["network_id", "noise_type", "noise_level"], inplace=True)

# Reset index after sorting
df_final.reset_index(drop=True, inplace=True)

# Save the final aggregated data to a new CSV
output_filename = "aggregated_fiedler_results.csv"
df_final.to_csv(output_filename, index=False)

print(f"\nAggregation complete! Saved to {output_filename}")
print("\nPreview of the aggregated DataFrame:")
print(df_final.head(15).to_string())

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_robustness_results(df, networks_to_plot):
    """
    Plots the algebraic connectivity vs. noise percentage for a specific list of networks.
    
    Parameters:
    df (pd.DataFrame): The aggregated dataframe.
    networks_to_plot (list): List of network_ids to plot (e.g., ['ppi', 'ppi_er', 'ppi_conf']).
    """
    
    # Filter the dataframe to only include the requested networks
    df_filtered = df[df['network_id'].isin(networks_to_plot)]
    
    if df_filtered.empty:
        print("None of the specified networks were found in the DataFrame.")
        return
    
    # Set up a grid for the subplots (2 columns max)
    n_plots = len(networks_to_plot)
    cols = 2 if n_plots > 1 else 1
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    
    # Use seaborn for standard pretty styling
    sns.set_theme(style="whitegrid")
    
    # Map your target types to specific colors and readable labels
    target_styles = {
        "perturbed_random_target": {"color": "#1f77b4", "label": "Random"},
        "perturbed_hub_target": {"color": "#ff7f0e", "label": "Hubs"},
        "perturbed_periphery_target": {"color": "#2ca02c", "label": "Periphery"}
    }
    
    for i, net_id in enumerate(networks_to_plot):
        ax = axes[i]
        df_net = df_filtered[df_filtered['network_id'] == net_id]
        
        if df_net.empty:
            ax.set_title(f"No data for: {net_id}")
            continue
            
        # 1. Grab baseline value for this network to draw a horizontal reference line
        baseline_row = df_net[df_net['noise_type'] == 'baseline']
        if not baseline_row.empty:
            base_val = baseline_row['avg_algebraic_connectivity'].values[0]
            # Horizontal line showing the baseline Fiedler value across the whole plot
            ax.axhline(base_val, color='gray', linestyle=':', alpha=0.7, label='Baseline Connectivity')
        
        # 2. Plot the perturbed lines with their standard deviation bands
        for p_type, style in target_styles.items():
            # Filter and sort so the lines draw left-to-right correctly
            df_pert = df_net[df_net['noise_type'] == p_type].sort_values('noise_level')
            
            if df_pert.empty:
                continue
                
            x = df_pert['noise_level']
            y = df_pert['avg_algebraic_connectivity']
            std = df_pert['std_algebraic_connectivity']
            
            # Draw the mean line
            ax.plot(x, y, label=style['label'], color=style['color'], marker='o', markersize=4)
            # Shade the standard deviation
            ax.fill_between(x, y - std, y + std, color=style['color'], alpha=0.2)
            
        # 3. Add the vertical marker for the Baseline (100% edges)
        ax.axvline(100, color='red', linestyle='--', alpha=0.8, label='Base Graph (100%)')
        
        # 4. Clean up labels and title
        ax.set_title(f"Robustness Profile: {net_id}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Network Size / Noise Level (%)", fontsize=12)
        ax.set_ylabel("Algebraic Connectivity (Fiedler Value)", fontsize=12)
        
        # Put the legend nicely outside the plot area or strictly in the corner
        ax.legend(loc='best', frameon=True)
        
    # Remove any completely empty subplots if your list length is odd
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# df = pd.read_csv("aggregated_fiedler_results.csv")
plot_robustness_results(df_final, ["ppi", "ppi_er", "ppi_conf", "ppi_sbm"])
# %%
