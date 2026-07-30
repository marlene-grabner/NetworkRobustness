import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------
# Function
# ---------------------------------------------------------

def plot_algorithm_ari_robustness(df: pd.DataFrame, algorithm_name: str, col_to_plot: str = 'vs_baseline_ari_mean', y_axis_label: str = 'ARI vs Base', view: str = 'all', save_fig: str = None):
    """
    Plots a 4x4 grid showing the ARI robustness (vs baseline) for all 16 networks 
    for ONE specific algorithm.
    
    Parameters:
    - view: 'all' (default), 'removal' (<=100%), or 'addition' (>=100%).
    """
    
    # 2. Calculate the relative network size for the X-axis
    def calc_relative_size(row):
        if row['type_of_noise'] == 'none' or row['level'] == 0:
            return 100.0
            
        level_pct = row['level'] * 100
        
        if row['type_of_noise'] == 'removal':
            return 100.0 - level_pct
        elif row['type_of_noise'] == 'addition':
            return 100.0 + level_pct
        return 100.0
        
    df['relative_size'] = df.apply(calc_relative_size, axis=1)
    
    # 3. Define the 4x4 Grid Layout
    grid_layout = [
        ['astro', 'astro_conf', 'astro_er', 'astro_sbm'],
        ['ppi', 'ppi_conf', 'ppi_er', 'ppi_sbm'],
        ['power', 'power_conf', 'power_er', 'power_sbm'],
        ['wiki', 'wiki_conf', 'wiki_er', 'wiki_sbm']
    ]
    
    row_labels = ['Astro', 'PPI', 'Power', 'Wiki']
    col_labels = ['Base Network', 'Configuration Model', 'Erdos-Renyi Model', 'Stoch. Block Model']
    
    # 4. Color and Style Mappings
    family_colors = {
        'ppi':   {'random': '#DA94A3', 'hub': '#E9204B', 'periphery': '#782235'}, 
        'astro': {'random': "#83E3B8", 'hub': "#089B3B", 'periphery': "#1F4836"}, 
        'power': {'random': "#BDA8F4", 'hub': '#5524E8', 'periphery': '#372278'}, 
        'wiki':  {'random': "#F4B266", 'hub': '#F6E825', 'periphery': "#925A04"}  
    }
    
    target_styles = {
        "random": {"ls": ":", "marker": "o", "label": "Random", "shade": "random"},
        "hub": {"ls": "-", "marker": "s", "label": "Hubs", "shade": "hub"},
        "periphery": {"ls": "--", "marker": "^", "label": "Periphery", "shade": "periphery"}
    }
    
    # 5. Figure Setup
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.15, wspace=0.1, top=0.92, bottom=0.1)
    
    plt.suptitle(f"{algorithm_name.title()}", fontsize=24, fontweight='bold')
    
    #tick_formatter = FuncFormatter(lambda val, pos: f"{val:g}%")
    tick_formatter = FuncFormatter(lambda val, pos: f"{int(val)}")

    # Filter ticks based on the selected view
    full_x_ticks = [50, 60, 70, 80, 100, 125, 150, 200, 250, 300]

    if view == 'removal':
        custom_x_ticks = [t for t in full_x_ticks if t <= 100]
    elif view == 'addition':
        custom_x_ticks = [t for t in full_x_ticks if t >= 100]
    else:
        custom_x_ticks = full_x_ticks
    
    # 6. Plotting Loop
    for row_idx in range(4):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            net_id = grid_layout[row_idx][col_idx]
            
            fam = row_labels[row_idx].lower()
            palette = family_colors.get(fam, family_colors['power'])
            
            df_net = df[df['base_network_name'] == net_id]
            
            if not df_net.empty:
                baseline_row = df_net[df_net['type_of_noise'] == 'none'].copy()
                
                for t_name, style in target_styles.items():
                    df_pert = df_net[df_net['target'] == t_name].copy()
                    
                    if not baseline_row.empty:
                        base_point = baseline_row.copy()
                        base_point['relative_size'] = 100.0
                        df_pert = pd.concat([df_pert, base_point])
                        
                    df_pert = df_pert.sort_values('relative_size')
                    
                    # Filter data points based on the view
                    if view == 'removal':
                        df_pert = df_pert[df_pert['relative_size'] <= 100]
                    elif view == 'addition':
                        df_pert = df_pert[df_pert['relative_size'] >= 100]
                    
                    if not df_pert.empty:
                        x = df_pert['relative_size']
                        y = df_pert[col_to_plot]
                        
                        std_col = col_to_plot.replace('_mean', '_std')
                        std = df_pert[std_col]
                        c = palette[style['shade']]
                        
                        ax.plot(x, y, color=c, linestyle=style['ls'], marker=style['marker'], 
                                markersize=5, linewidth=2)
                        
                        if 'ari' in col_to_plot:
                            y_lower = np.maximum(0, y - std)
                            y_upper = np.minimum(1.0, y + std)
                        else:
                            y_lower = y - std
                            y_upper = y + std
                            
                        ax.fill_between(x, y_lower, y_upper, color=c, alpha=0.15)
            
            # Formatting constraints
            ax.set_xscale('log')
            ax.set_xticks(custom_x_ticks)
            ax.set_xticks([], minor=True)
            ax.xaxis.set_major_formatter(tick_formatter)
            
            # Lock the X-axis limits to prevent whitespace
            if view == 'removal':
                ax.set_xlim(right=100)
            elif view == 'addition':
                ax.set_xlim(left=100)
            
            if 'ari' in col_to_plot:
                ax.set_ylim(-0.05, 1.05)
            
            ax.axvline(100, color='gray', linestyle='-', alpha=0.3, linewidth=2)
            
            # Label Row & Column Headers
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=16, fontweight='bold', pad=15)
            if col_idx == 0:
                ax.set_ylabel(f"{row_labels[row_idx]}\n{y_axis_label}", fontsize=14, fontweight='bold', labelpad=10)
            if row_idx == 3:
                ax.set_xlabel("Relative Network Size", fontsize=12, labelpad=10)
            
            # Grid and Spines
            ax.grid(True, which="major", axis="both", color="#E0E0E0", linestyle="--", alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color('#CCCCCC')

    # 7. Add a single neutral legend at the bottom
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2.5, 
               ls=style['ls'], marker=style['marker'], markersize=8, label=style['label'])
        for style in target_styles.values()
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False, 
               fontsize=16, bbox_to_anchor=(0.5, 0.02))
    
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_fig}")
        
    plt.show()

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------

input_data_path = "outputs/local_structure/overview_csvs/perturbed"
output_data_path = "outputs/local_structure/figures/perturbed"


df_louvain = pd.read_csv(f"{input_data_path}/louvain_aggregated.csv")
df_leiden = pd.read_csv(f"{input_data_path}/leiden_aggregated.csv")
df_infomap = pd.read_csv(f"{input_data_path}/infomap_aggregated.csv")
df_labelprop = pd.read_csv(f"{input_data_path}/label_propagation_aggregated.csv")


# Options for columns to plot: within_ari_mean ,vs_baseline_ari_mean,n_communities_mean
for algorithm in ['louvain', 'leiden', 'infomap', 'label_propagation']:
    df = pd.read_csv(f"{input_data_path}/{algorithm}_aggregated.csv")
    # View all
    plot_algorithm_ari_robustness(df, algorithm.title().replace("_", " "), col_to_plot='vs_baseline_ari_mean', y_axis_label='ARI vs Base', view = 'all', save_fig=f"{output_data_path}/{algorithm}_robustness_view_all.pdf")
    plot_algorithm_ari_robustness(df, algorithm.title().replace("_", " "), col_to_plot='within_ari_mean', y_axis_label='Within ARI', view = 'all', save_fig=f"{output_data_path}/{algorithm}_within_ari_view_all.pdf")
    plot_algorithm_ari_robustness(df, algorithm.title().replace("_", " "), col_to_plot='n_communities_mean', y_axis_label='Number of Communities', view = 'all', save_fig=f"{output_data_path}/{algorithm}_n_communities_view_all.pdf")

    # View removal
    plot_algorithm_ari_robustness(df, algorithm.title().replace("_", " "), col_to_plot='vs_baseline_ari_mean', y_axis_label='ARI vs Base', view = 'removal', save_fig=f"{output_data_path}/{algorithm}_robustness_view_removal.pdf")
    plot_algorithm_ari_robustness(df, algorithm.title().replace("_", " "), col_to_plot='within_ari_mean', y_axis_label='Within ARI', view = 'removal', save_fig=f"{output_data_path}/{algorithm}_within_ari_view_removal.pdf")
    plot_algorithm_ari_robustness(df, algorithm.title().replace("_", " "), col_to_plot='n_communities_mean', y_axis_label='Number of Communities', view = 'removal', save_fig=f"{output_data_path}/{algorithm}_n_communities_view_removal.pdf")

    # View addition
    plot_algorithm_ari_robustness(df, algorithm.title().replace("_", " "), col_to_plot='vs_baseline_ari_mean', y_axis_label='ARI vs Base', view = 'addition', save_fig=f"{output_data_path}/{algorithm}_robustness_view_addition.pdf")
    plot_algorithm_ari_robustness(df, algorithm.title().replace("_", " "), col_to_plot='within_ari_mean', y_axis_label='Within ARI', view = 'addition', save_fig=f"{output_data_path}/{algorithm}_within_ari_view_addition.pdf")
    plot_algorithm_ari_robustness(df, algorithm.title().replace("_", " "), col_to_plot='n_communities_mean', y_axis_label='Number of Communities', view = 'addition', save_fig=f"{output_data_path}/{algorithm}_n_communities_view_addition.pdf")
