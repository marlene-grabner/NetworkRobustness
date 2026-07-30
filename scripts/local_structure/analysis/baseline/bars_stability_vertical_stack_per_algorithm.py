import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_algorithm_stability_subplots(df: pd.DataFrame, save_fig: str = None):
    
    # 1. Clean and normalize algorithm names
    df['algorithm'] = df['algorithm'].str.replace('.npz', '', regex=False)
    df['algorithm'] = df['algorithm'].replace({
        'label_propagation': 'Label Prop',
        'Label_Propagation': 'Label Prop',
        'louvain': 'Louvain',
        'leiden': 'Leiden',
        'infomap': 'Infomap'
    })
    df['algorithm'] = df['algorithm'].str.title()
    df['algorithm'] = df['algorithm'].replace('Label Prop', 'Label Prop')
    
    # 2. Normalize network names
    df['network'] = df['network'].str.replace('_config', '_conf')
    
    # 3. Define the exact colors and nice labels
    color_map = {
        'power': "#122EC8", 'power_conf': "#3C76E9", 'power_er': "#67BFF1", 'power_sbm': "#9DD2F0",
        'ppi': "#A43531", 'ppi_conf': "#D43838", 'ppi_er': "#F77471", 'ppi_sbm': "#F2A7A7",
        'astro': "#02800C", 'astro_conf': "#21C02E", 'astro_er': "#4ED97F", 'astro_sbm': "#A4E4AB",
        'wiki': "#F3B311", 'wiki_conf': "#EACE40", 'wiki_er': "#FFF34C", 'wiki_sbm': "#F4F299"
    }
    
    label_map = {
        'power': 'Western US Power Grid', 'power_conf': 'Config Model (Power)', 'power_er': 'Erdos Renyi (Power)', 'power_sbm': 'Stoch. Block Model (Power)',
        'ppi': 'Protein Interaction Network', 'ppi_conf': 'Config Model (PPI)', 'ppi_er': 'Erdos Renyi (PPI)', 'ppi_sbm': 'Stoch. Block Model (PPI)',
        'astro': 'Astrophysics Collaboration', 'astro_conf': 'Config Model (Astro)', 'astro_er': 'Erdos Renyi (Astro)', 'astro_sbm': 'Stoch. Block Model (Astro)',
        'wiki': 'Wikipedia Vote', 'wiki_conf': 'Config Model (Wiki)', 'wiki_er': 'Erdos Renyi (Wiki)', 'wiki_sbm': 'Stoch. Block Model (Wiki)'
    }
    
    # 4. Define groups to create visual gaps
    groups_raw = [
        ['astro', 'astro_conf', 'astro_er', 'astro_sbm'],
        ['ppi', 'ppi_conf', 'ppi_er', 'ppi_sbm'],
        ['power', 'power_conf', 'power_er', 'power_sbm'],
        ['wiki', 'wiki_conf', 'wiki_er', 'wiki_sbm']
    ]
    
    algorithms = ['Infomap', 'Leiden', 'Louvain', 'Label Prop']
    
    # 5. Set up the 1x4 Figure
    fig, axes = plt.subplots(1, 4, figsize=(22, 10), sharey=True)
    # wspace to 0.25 to clearly separate the algorithms
    fig.subplots_adjust(wspace=0.25, top=0.88, bottom=0.1) 
    
    fig.text(0.5, 0.02, 'Mean Adjusted Rand Index', ha='center', fontsize=16, fontweight='bold')
    
    for idx, alg in enumerate(algorithms):
        ax = axes[idx]
        
        y_positions = []
        labels = []
        widths = []
        err_low = []
        err_high = []
        colors = []
        
        current_y = 0
        
        for group in groups_raw:
            for net_raw in group:
                y_positions.append(current_y)
                labels.append(label_map.get(net_raw, net_raw))
                colors.append(color_map.get(net_raw, '#333333'))
                
                row = df[(df['network'] == net_raw) & (df['algorithm'] == alg)]
                if not row.empty:
                    val = row['ari_mean'].values[0]
                    std = row['ari_std'].values[0]
                    if pd.isna(std): std = 0
                    
                    widths.append(val)
                    err_low.append(min(val, std))
                    err_high.append(min(1.0 - val, std))
                else:
                    widths.append(0)
                    err_low.append(0)
                    err_high.append(0)
                    
                current_y += 1
            # Add physical space after each family group
            current_y += 1.5 
            
        # REDUCED height to 0.55 for sleeker, less clunky bars
        # Thinner error bars to match the elegant style
        ax.barh(y_positions, widths, height=0.65, color=colors, zorder=3,
                xerr=[err_low, err_high], 
                error_kw=dict(lw=1.2, capsize=3.0, capthick=1.2, ecolor='#444444', alpha=0.8))
               
        # Subplot Formatting
        ax.set_title(alg, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlim(0, 1.05)
        
        # Explicitly set clean x-ticks so the grid looks uniform across all 4 plots
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        
        if idx == 0:
            ax.invert_yaxis()
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=13)
        
        # Remove top and right spines, but KEEP left and bottom in a soft gray to anchor the plots
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('#B0B0B0')
            ax.spines[spine].set_linewidth(1.2)
            
        # Vertical gridlines (underneath bars due to zorder)
        ax.grid(True, axis='x', linestyle='--', color='#E0E0E0', alpha=1.0, zorder=0)
        ax.tick_params(axis='x', labelsize=12, colors='#333333')
        
        # Hide the Y-axis tick marks
        ax.tick_params(axis='y', length=0)

    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_fig}")
        
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


plot_algorithm_stability_subplots(df, save_fig=f'{output_folder}/baseline_community_stability_horizontal_bar_charts.pdf')

