import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# --- Plotting Function ---
# ===========================================================================


def plot_baseline_fiedler(df, save_fig = None):
    """
    Plots the baseline algebraic connectivity (Fiedler value) for all 16 networks.
    Groups them by family (Base, Config, ER, SBM) with a visual gap between families.
    """
    # 1. Filter for baseline data only
    df_base = df[df['noise_type'] == 'baseline'].copy()
    
    if df_base.empty:
        print("No baseline data found in the DataFrame.")
        return

    # 2. Define the exact grouping order
    # (Checking for both _config and _conf depending on how they are named in your df)
    plot_groups = [
        ['power', 'power_conf', 'power_config', 'power_er', 'power_sbm'],
        ['ppi', 'ppi_conf', 'ppi_config', 'ppi_er', 'ppi_sbm'],
        ['astro', 'astro_conf', 'astro_config', 'astro_er', 'astro_sbm'],
        ['wiki', 'wiki_conf', 'wiki_config', 'wiki_er', 'wiki_sbm']
    ]

    # 3. Incorporate your exact colors
    color_map = {
        'power': '#372278', 'power_config': "#5524E8", 'power_conf': "#5524E8", 'power_er': "#6A6085", 'power_sbm': "#C8BEE3",
        'ppi': '#782235', 'ppi_config': '#DA94A3', 'ppi_conf': '#DA94A3', 'ppi_er': "#E9204B", 'ppi_sbm': '#1C0006',
        'astro': '#227851', 'astro_config': "#BDF0D9", 'astro_conf': "#BDF0D9", 'astro_er': "#16C553", 'astro_sbm': "#486055",
        'wiki': '#E8AD0C', 'wiki_config': "#474501", 'wiki_conf': "#474501", 'wiki_er': "#F6E825", 'wiki_sbm': "#F1DDA6"
    }

    # 4. Label dictionary
    label_map = {
        'power': 'Western US Power Grid',
        'power_config': 'Config Model (Power)', 'power_conf': 'Config Model (Power)',
        'power_er': 'Erdos Renyi (Power)',
        'power_sbm': 'Stoch. Block Model (Power)',
        'ppi': 'Protein Interaction Network',
        'ppi_config': 'Config Model (PPI)', 'ppi_conf': 'Config Model (PPI)',
        'ppi_er': 'Erdos Renyi (PPI)',
        'ppi_sbm': 'Stoch. Block Model (PPI)',
        'astro': 'Astrophysics Collaboration',
        'astro_config': 'Config Model (Astro)', 'astro_conf': 'Config Model (Astro)',
        'astro_er': 'Erdos Renyi (Astro)',
        'astro_sbm': 'Stoch. Block Model (Astro)',
        'wiki': 'Wikipedia Vote',
        'wiki_config': 'Config Model (Wiki)', 'wiki_conf': 'Config Model (Wiki)',
        'wiki_er': 'Erdos Renyi (Wiki)',
        'wiki_sbm': 'Stoch. Block Model (Wiki)'
    }

    # 5. Extract data and calculate X positions to create gaps between families
    x_positions = []
    heights = []
    colors = []
    labels = []
    
    current_x = 0
    for group in plot_groups:
        added_in_group = False
        for net_id in group:
            val_row = df_base[df_base['network_id'] == net_id]
            if not val_row.empty:
                # Get the baseline fiedler value
                heights.append(val_row['avg_algebraic_connectivity'].values[0])
                x_positions.append(current_x)
                colors.append(color_map.get(net_id, '#333333')) # fallback color
                labels.append(label_map.get(net_id, net_id))
                current_x += 1
                added_in_group = True
        
        # Add a gap after each family group if we actually plotted something
        if added_in_group:
            current_x += 1.2  

    # 6. Set up the figure
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.subplots_adjust(bottom=0.35) # Make room for rotated labels
    
    # 7. Plot the bars
    bars = ax.bar(x_positions, heights, color=colors, edgecolor='none', width=0.85)
    
    # 8. Formatting
    ax.set_ylabel("Fiedler Value", fontsize=14, labelpad=15)
    
    # Format X-axis ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    
    # Grid and Spines
    ax.set_axisbelow(True) # Put grid behind bars
    ax.grid(True, axis="y", color="#E0E0E0", linestyle="-", alpha=0.7)
    ax.grid(False, axis="x") # Hide vertical grid lines
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Optional: Add the exact values on top of the bars if they are very small
    for bar in bars:
        yval = bar.get_height()
        # Format to 4 decimal places so small values are readable
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(heights)*0.01), 
                f'{yval:.4f}', ha='center', va='bottom', fontsize=10, rotation=0, color='dimgray')
        
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')

    plt.show()


# ============================================================================
# --- Reading in the dataframe ---
# ===========================================================================

df = pd.read_csv("outputs/global_properties/algebraic_connectivity/aggregated_mean_fiedler_values_perturbed_plus_baseline.csv")

# ============================================================================
# --- Plotting ---
# ===========================================================================
output_folder_figures = "outputs/global_properties/algebraic_connectivity/figures"

plot_baseline_fiedler(df, save_fig=f"{output_folder_figures}/baseline_fiedler_values.pdf")



