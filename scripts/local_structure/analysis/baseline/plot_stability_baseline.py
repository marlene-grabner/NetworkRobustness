import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm, SymLogNorm


# ============================================================================
# --- Plotting functions ---
# ===========================================================================

def plot_community_stats_dashboard(df: pd.DataFrame, save_fig: str = None):
    
    # Clean up algorithm names for the plot
    df['algorithm'] = df['algorithm'].str.replace('.npz', '', regex=False)
    df['algorithm'] = df['algorithm'].replace({
        'label_propagation': 'Label Prop.',
        'louvain': 'Louvain',
        'leiden': 'Leiden',
        'infomap': 'Infomap'
    })
    
    # Network label mapping
    label_map = {
        'power': 'Western US Power Grid', 'power_conf': 'Config Model (Power)', 'power_er': 'Erdos Renyi (Power)', 'power_sbm': 'Stoch. Block Model (Power)',
        'ppi': 'Protein Interaction Network', 'ppi_conf': 'Config Model (PPI)', 'ppi_er': 'Erdos Renyi (PPI)', 'ppi_sbm': 'Stoch. Block Model (PPI)',
        'astro': 'Astrophysics Collaboration', 'astro_conf': 'Config Model (Astro)', 'astro_er': 'Erdos Renyi (Astro)', 'astro_sbm': 'Stoch. Block Model (Astro)',
        'wiki': 'Wikipedia Vote', 'wiki_conf': 'Config Model (Wiki)', 'wiki_er': 'Erdos Renyi (Wiki)', 'wiki_sbm': 'Stoch. Block Model (Wiki)'
    }
    
    # Apply nice names, fallback to original if not found
    df['network_nice'] = df['network'].apply(lambda x: label_map.get(x, x.replace('_', ' ').title()))
    
    # Define the exact explicit order for the rows
    desired_order = [
        'Astrophysics Collaboration', 'Config Model (Astro)', 'Erdos Renyi (Astro)', 'Stoch. Block Model (Astro)',
        'Protein Interaction Network', 'Config Model (PPI)', 'Erdos Renyi (PPI)', 'Stoch. Block Model (PPI)',
        'Western US Power Grid', 'Config Model (Power)', 'Erdos Renyi (Power)', 'Stoch. Block Model (Power)',
        'Wikipedia Vote', 'Config Model (Wiki)', 'Erdos Renyi (Wiki)', 'Stoch. Block Model (Wiki)'
    ]
    
    # 2. Create pivots for the heatmaps
    metrics = {
        'ari_mean': ('Mean ARI', 'viridis'),
        'num_comms_mean': ('Number of Communities', 'mako'),
        'avg_size_mean': ('Average Community Size', 'rocket'),
        'singletons_mean': ('Number of Singletons', 'flare')
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.subplots_adjust(hspace=0.4, wspace=0.6)
    axes = axes.flatten()
    
    for i, (col, (title, cmap)) in enumerate(metrics.items()):
        ax = axes[i]
        
        # Pivot the dataframe to get a Matrix of Networks (Rows) x Algorithms (Columns)
        pivot_df = df.pivot(index='network_nice', columns='algorithm', values=col)
        
        # Apply the custom sort
        valid_order = [net for net in desired_order if net in pivot_df.index]
        pivot_df = pivot_df.reindex(valid_order)
        
        # --- NEW: Apply Logarithmic Scaling where appropriate ---
        norm = None
        if col in ['num_comms_mean', 'avg_size_mean']:
            min_val = pivot_df.min().min()
            # If we have 0s (common for singletons), standard LogNorm fails. 
            # SymLogNorm handles 0s safely by switching to linear near 0.
            if min_val <= 0:
                norm = SymLogNorm(linthresh=1, vmin=0, vmax=pivot_df.max().max())
            else:
                norm = LogNorm(vmin=min_val, vmax=pivot_df.max().max())
                
        sns.heatmap(pivot_df, ax=ax, cmap=cmap, annot=True, fmt=".2f" if 'ari' in col else ".0f",
                    linewidths=.5, cbar_kws={"shrink": .8}, norm=norm)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.tick_params(axis='y', rotation=0, labelsize=11)
            
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
        
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


plot_community_stats_dashboard(df, save_fig=f'{output_folder}/baseline_community_stats_dashboard.pdf')