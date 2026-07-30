import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_algorithm_stability_barchart(df: pd.DataFrame, save_fig: str = None):
    
    # 1. Clean algorithm names robustly
    df['algorithm'] = df['algorithm'].str.replace('.npz', '', regex=False)
    df['algorithm'] = df['algorithm'].replace({
        'label_propagation': 'Label Prop',
        'Label_Propagation': 'Label Prop',
        'louvain': 'Louvain',
        'leiden': 'Leiden',
        'infomap': 'Infomap'
    })
    # Title case them just in case
    df['algorithm'] = df['algorithm'].str.title()
    df['algorithm'] = df['algorithm'].replace('Label Prop', 'Label Prop')
    
    # 2. Network label mapping
    label_map = {
        'power': 'Western US Power Grid', 'power_conf': 'Config Model (Power)', 'power_er': 'Erdos Renyi (Power)', 'power_sbm': 'Stoch. Block Model (Power)',
        'ppi': 'Protein Interaction Network', 'ppi_conf': 'Config Model (PPI)', 'ppi_er': 'Erdos Renyi (PPI)', 'ppi_sbm': 'Stoch. Block Model (PPI)',
        'astro': 'Astrophysics Collaboration', 'astro_conf': 'Config Model (Astro)', 'astro_er': 'Erdos Renyi (Astro)', 'astro_sbm': 'Stoch. Block Model (Astro)',
        'wiki': 'Wikipedia Vote', 'wiki_conf': 'Config Model (Wiki)', 'wiki_er': 'Erdos Renyi (Wiki)', 'wiki_sbm': 'Stoch. Block Model (Wiki)'
    }
    
    df['network_nice'] = df['network'].apply(lambda x: label_map.get(x, x.replace('_', ' ')))
    
    # 3. Define a much nicer, cohesive color palette
    algo_colors = {
        'Infomap': '#EC9192',      # Warm Coral
        'Leiden': '#DFBE99',       # Muted Teal
        'Louvain': '#B5BD89',      # Vibrant Blue
        'Label Prop': '#729EA1'    # Deep Purple
    }
    
    # 4. Explicitly group the networks to create gaps
    groups = [
        ['Astrophysics Collaboration', 'Config Model (Astro)', 'Erdos Renyi (Astro)', 'Stoch. Block Model (Astro)'],
        ['Protein Interaction Network', 'Config Model (PPI)', 'Erdos Renyi (PPI)', 'Stoch. Block Model (PPI)'],
        ['Western US Power Grid', 'Config Model (Power)', 'Erdos Renyi (Power)', 'Stoch. Block Model (Power)'],
        ['Wikipedia Vote', 'Config Model (Wiki)', 'Erdos Renyi (Wiki)', 'Stoch. Block Model (Wiki)']
    ]
    
    algorithms = ['Infomap', 'Leiden', 'Louvain', 'Label Prop']
    
    # 5. Extract and format data for manual plotting
    x_positions = []
    labels = []
    means = {alg: [] for alg in algorithms}
    err_low = {alg: [] for alg in algorithms}
    err_high = {alg: [] for alg in algorithms}
    
    current_x = 0
    for group in groups:
        for net in group:
            labels.append(net)
            x_positions.append(current_x)
            
            net_df = df[df['network_nice'] == net]
            
            for alg in algorithms:
                alg_row = net_df[net_df['algorithm'] == alg]
                if not alg_row.empty:
                    val = alg_row['ari_mean'].values[0]
                    std = alg_row['ari_std'].values[0]
                    if pd.isna(std): std = 0
                    
                    means[alg].append(val)
                    # FIX: Cap error bars so they never go below 0 or above 1.0
                    err_low[alg].append(min(val, std)) 
                    err_high[alg].append(min(1.0 - val, std))
                else:
                    means[alg].append(0)
                    err_low[alg].append(0)
                    err_high[alg].append(0)
            
            current_x += 1
        # Add extra physical offset space after each family
        current_x += 1.5 
        
    # 6. Set up Figure
    fig, ax = plt.subplots(figsize=(18, 7))
    
    # Calculate bar widths and offsets
    n_algs = len(algorithms)
    total_width = 0.8
    bar_width = total_width / n_algs
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, n_algs)
    
    # Plot Bars
    for i, alg in enumerate(algorithms):
        pos = np.array(x_positions) + offsets[i]
        y = means[alg]
        yerr = [err_low[alg], err_high[alg]]
        
        ax.bar(pos, y, width=bar_width, label=alg, color=algo_colors.get(alg, '#333'), zorder=3,
               yerr=yerr, error_kw=dict(lw=1.2, capsize=2.5, capthick=1.2, ecolor='#444444', alpha=0.8))
    
    # 7. Aesthetics & Formatting
    ax.set_ylabel("Mean Adjusted Rand Index", fontsize=14, labelpad=15)
    ax.set_ylim(0, 1.05)
    
    # Format X-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    
    # Remove ALL spines
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Nice horizontal grid
    ax.grid(True, axis='y', linestyle='-', color='#E8E8E8', alpha=0.8, zorder=0)
    
    # Place Legend at the top, centered, 4 columns
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=14)
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight', dpi=300)
    plt.show()

# ============================================================================
# --- Execute ---
# ===========================================================================

output_folder = "outputs/local_structure/figures/baseline"
input_data_path = "outputs/local_structure/overview_csvs/baseline"

df_infomap = pd.read_csv(f"{input_data_path}/baseline_community_stats_infomap.csv")
df_louvain = pd.read_csv(f"{input_data_path}/baseline_community_stats_louvain.csv")
df_leiden = pd.read_csv(f"{input_data_path}/baseline_community_stats_leiden.csv")
df_label_prop = pd.read_csv(f"{input_data_path}/baseline_community_stats_label_propagation.csv")

df = pd.concat([df_infomap, df_louvain, df_leiden, df_label_prop], ignore_index=True)

plot_algorithm_stability_barchart(df, save_fig=f"{output_folder}/algorithm_stability_barchart_grouped_by_network.pdf") 