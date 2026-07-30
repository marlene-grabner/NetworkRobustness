#%%
import os, re, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

######################
#   Functions
######################

# ===================================================
# Extract features (network name, action, noise level) from network_id
# ===================================================

def extract_features(network_id):
    """
    Extracts the network name, action, and noise level from the network_id string.
    """
    net_id_str = str(network_id)
    
    # 1. Extract Network Name (Base + Optional Variation)
    # This looks for your 4 specific base names, optionally followed by _config, _er, or _sbm
    network_pattern = r'(western_us_power_grid|chloe_ppi_lcc_2026_02_23|ca-AstroPH_gcc|wiki-Vote_gcc)(?:_config|_er|_sbm)?'
    network_match = re.search(network_pattern, net_id_str)
    network = network_match.group(0) if network_match else 'unknown_network'
    
    # 2. Extract Action (Addition or Removal)
    if re.search(r'(addition|added)', net_id_str, re.IGNORECASE):
        action = 'addition'
    elif re.search(r'(removal|removed)', net_id_str, re.IGNORECASE):
        action = 'removal'
    else:
        action = 'none' # Usually applies to baseline
        
    # 3. Extract Noise Level
    # Looks for "noise_XpY" and converts the "p" to a decimal point
    noise_match = re.search(r'noise_(\d+p\d+|\d+)', net_id_str)
    if noise_match:
        noise_level = float(noise_match.group(1).replace('p', '.'))
    else:
        noise_level = 0.0 # Baseline has no noise
        
    return pd.Series([network, action, noise_level])

def interquartile_range(x):
    """Calculates the IQR, which is the median's equivalent to standard deviation."""
    return x.quantile(0.75) - x.quantile(0.25)


# ===================================================
# Plotting function for GCC
# ===================================================


def plot_gcc_robustness(df_agg, networks, color_dict, measure = 'gcc', metric='mean', noise_type_filter=None, save_fig=None):
    """
    Plots the GCC robustness curve for specified networks, focusing only on edge/node removal.
    
    Parameters:
    * df_agg: The aggregated pandas DataFrame.
    * networks: List of strings specifying which networks to plot.
    * color_dict: Dictionary mapping network names to standard matplotlib colors.
    * measure: String, either 'gcc' or 'num_singletons'. Dictates which measure to plot.
    * metric: String, either 'mean' or 'median'. Dictates the central tendency and spread.
    * noise_type_filter: Optional string. Filters for a specific noise_type.
    * save_fig: Optional string. If provided, saves the figure to this path (e.g., 'plot.pdf').
    """
    
    # Dictionary to rename networks for the legend
    label_map = {
        'western_us_power_grid': 'Western US Power Grid',
        'western_us_power_grid_config': 'Configuration Model (Power)',
        'western_us_power_grid_er': 'Erdos Renyi (Power)',
        'western_us_power_grid_sbm': 'Stochastic Block Model (Power)',
        
        'chloe_ppi_lcc_2026_02_23': 'Protein Interaction Network',
        'chloe_ppi_lcc_2026_02_23_config': 'Configuration Model (PPI)',
        'chloe_ppi_lcc_2026_02_23_er': 'Erdos Renyi (PPI)',
        'chloe_ppi_lcc_2026_02_23_sbm': 'Stochastic Block Model (PPI)',
        
        'ca-AstroPH_gcc': 'Astrophysics Collaboration',
        'ca-AstroPH_gcc_config': 'Configuration Model (Astro)',
        'ca-AstroPH_gcc_er': 'Erdos Renyi (Astro)',
        'ca-AstroPH_gcc_sbm': 'Stochastic Block Model (Astro)',
        
        'wiki-Vote_gcc': 'Wikipedia Vote',
        'wiki-Vote_gcc_config': 'Configuration Model (Wiki)',
        'wiki-Vote_gcc_er': 'Erdos Renyi (Wiki)',
        'wiki-Vote_gcc_sbm': 'Stochastic Block Model (Wiki)'
    }

    # 1. Base Filter: Only removals, and only the networks we asked for
    plot_df = df_agg[(df_agg['action'] == 'removal') & (df_agg['network'].isin(networks))].copy()
    
    if noise_type_filter:
        plot_df = plot_df[plot_df['noise_type'] == noise_type_filter]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 2. Iterate and Plot
    for network in networks:
        net_data = plot_df[plot_df['network'] == network].sort_values(by='noise_level')
        
        if net_data.empty:
            continue
            
        x = net_data['noise_level'].values
        
        # 3. Determine measure and metric logic
        if measure == 'gcc':
            if metric == 'mean':
                y = net_data['gcc_mean'].values
                spread = net_data['gcc_std'].values
                lower_bound = y - spread
                upper_bound = y + spread
            elif metric == 'median':
                y = net_data['gcc_median'].values
                half_iqr = net_data['gcc_iqr'].values / 2.0
                lower_bound = y - half_iqr
                upper_bound = y + half_iqr
            ylabel = 'GCC Size (Mean ± Std)'
        elif measure == 'num_singletons':
            if metric == 'mean':
                y = net_data['num_singletons_mean'].values
                spread = net_data['num_singletons_std'].values
                lower_bound = y - spread
                upper_bound = y + spread
            elif metric == 'median':
                y = net_data['num_singletons_median'].values
                half_iqr = net_data['num_singletons_iqr'].values / 2.0
                lower_bound = y - half_iqr
                upper_bound = y + half_iqr
            ylabel = 'Number of Singletons (Mean ± Std)'
            
            
        else:
            raise ValueError("Measure must be either 'gcc' or 'num_singletons', and metric must be either 'mean' or 'median'")
            
        lower_bound = np.clip(lower_bound, 0.0, 1.0)
        upper_bound = np.clip(upper_bound, 0.0, 1.0)
        
        # 4. Draw the elements
        color = color_dict.get(network, 'black') 
        
        # Get the pretty name for the legend, fallback to raw name if missing
        legend_label = label_map.get(network, network)
        
        ax.plot(x, y, color=color, label=legend_label, linewidth=2, marker='o', markersize=5)
        ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2)

    # 5. Formatting (Larger fonts, no bolding, no title)
    ax.set_xlabel("Noise Level (Fraction Removed)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Larger tick fonts
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Position legend at the top, spread into columns
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, fontsize=12)
    
    # Calculate tight bounding box so legend isn't clipped
    plt.tight_layout()
    
    # 6. Save option
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
        

#%%

####################################################################################
# Parameters
output_df_path = './outputs/global_properties/aggregated_gcc_singletons.csv'
output_figures_path = './outputs/global_properties/figures/gcc_singletons/'

######################
#   Analysis
######################

# ===================================================
# Make dataframe of GCC and singletons per noise level, action, and network
# ===================================================

# --- 1. Read and Combine all CSVs ---
folder_path = "./outputs/global_properties/gcc_singletons"
all_files = glob.glob(os.path.join(folder_path, "*.csv"))
print(f"Found files: {all_files}")

# Read all files and concatenate them into one large DataFrame
df_list = [pd.read_csv(f) for f in all_files]
df_raw = pd.concat(df_list, ignore_index=True)
print(f"Combined DataFrame of all files")
print(df_raw.head())

# --- 2. Extract New Columns ---
# Apply the extraction function to create the new columns
df_raw[['network', 'action', 'noise_level']] = df_raw['network_id'].apply(extract_features)

# Rename the perturbation column as requested
df_raw = df_raw.rename(columns={'perturbation_method': 'noise_type'})

print("After reshaping")
print(df_raw.head())

# --- 3. Aggregate the Repeats ---
# Group by our newly extracted identifiers
grouped = df_raw.groupby(['network', 'noise_type', 'action', 'noise_level'])

# Calculate mean, median, std, and IQR for the numeric columns
df_agg = grouped.agg({
    'num_singletons': ['mean', 'median', 'std', interquartile_range],
    'gcc': ['mean', 'median', 'std', interquartile_range]
}).reset_index()

print("After calculation of mean, median, std, and IQR")
print(df_agg.head())

# --- 4. Clean Up Column Names ---
# The aggregation creates MultiIndex columns (e.g., ('gcc', 'mean')). Let's flatten them.
df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]

# Rename the custom IQR function columns for clarity
df_agg = df_agg.rename(columns={
    'num_singletons_interquartile_range': 'num_singletons_iqr',
    'gcc_interquartile_range': 'gcc_iqr'
})

print("After cleaning up column names")
print(df_agg.head())

print(f"Saving the aggregated dataframe as CSV to: {output_df_path}")
df_agg.to_csv(output_df_path, index=False)

#%%
# ===================================================
# Plotting
# ===================================================

network_colors = {
    'western_us_power_grid': '#372278',
    'western_us_power_grid_config': "#5524E8",
    'western_us_power_grid_er': "#6A6085",
    'western_us_power_grid_sbm': "#C8BEE3",
    'chloe_ppi_lcc_2026_02_23': '#782235',
    'chloe_ppi_lcc_2026_02_23_config': '#DA94A3',
    'chloe_ppi_lcc_2026_02_23_er': "#E9204B",
    'chloe_ppi_lcc_2026_02_23_sbm': '#1C0006',
    'ca-AstroPH_gcc': '#227851',
    'ca-AstroPH_gcc_config': "#BDF0D9",
    'ca-AstroPH_gcc_er': "#16C553",
    'ca-AstroPH_gcc_sbm': "#486055",
    'wiki-Vote_gcc': '#E8AD0C',
    'wiki-Vote_gcc_config': "#474501",
    'wiki-Vote_gcc_er': "#F6E825",
    'wiki-Vote_gcc_sbm': "#F1DDA6"
}

power_networks = ['western_us_power_grid', 'western_us_power_grid_config', 'western_us_power_grid_er', 'western_us_power_grid_sbm']
ppi_networks = ['chloe_ppi_lcc_2026_02_23', 'chloe_ppi_lcc_2026_02_23_config', 'chloe_ppi_lcc_2026_02_23_er', 'chloe_ppi_lcc_2026_02_23_sbm']
collab_networks = ['ca-AstroPH_gcc', 'ca-AstroPH_gcc_config', 'ca-AstroPH_gcc_er', 'ca-AstroPH_gcc_sbm']
wiki_networks = ['wiki-Vote_gcc', 'wiki-Vote_gcc_config', 'wiki-Vote_gcc_er', 'wiki-Vote_gcc_sbm']

all_network_sets = [power_networks, ppi_networks, collab_networks, wiki_networks]
perturbation_types = ['perturbed_periphery_target', 'perturbed_hub_target', 'perturbed_random_target']

for network in all_network_sets:
    for perturbation in perturbation_types:
        for measure in ['gcc', 'num_singletons']:
            for metric in ['mean', 'median']:
                plot_gcc_robustness(
                    df_agg=df_agg, 
                    networks=network, 
                    color_dict=network_colors, 
                    measure=measure,
                    metric=metric,                  # Toggle to 'mean' if preferred
                    noise_type_filter=perturbation, # Optional
                    save_fig=f"{output_figures_path}/{network[0]}_{perturbation}_average_{measure}_{metric}.pdf"
                )

