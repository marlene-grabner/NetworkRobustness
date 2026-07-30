import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# =====================================================================
# Function
# =====================================================================

def plot_local_robustness(df: pd.DataFrame, algorithm: str, k_val: str = '25', metric: str = 'auprc', view: str = 'all', save_fig: str = None):
    """
    Plots a 4x4 grid showing local neighborhood robustness.
    Filters the massive DataFrame down to one Algorithm, one K, and plots one Metric.
    """
    
    # 1. Filter the data to exactly what we want to plot
    df_plot = df[(df['algorithm'] == algorithm) & (df['k_category'] == k_val)].copy()
    
    if df_plot.empty:
        print(f"No data found for algorithm '{algorithm}' at k='{k_val}'.")
        return
        
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    
    if mean_col not in df_plot.columns:
        print(f"Metric '{mean_col}' not found in DataFrame.")
        return

    # 2. Calculate relative size
    def calc_relative_size(row):
        if row['noise_level'] == 0:
            return 100.0
        
        level_pct = row['noise_level'] * 100
        
        if row['modification_type'] == 'removal':
            return 100.0 - level_pct
        elif row['modification_type'] == 'addition':
            return 100.0 + level_pct
        return 100.0
        
    df_plot['relative_size'] = df_plot.apply(calc_relative_size, axis=1)
    
    # 3. Grid Layout
    grid_layout = [
        ['astro', 'astro_conf', 'astro_er', 'astro_sbm'],
        ['ppi', 'ppi_conf', 'ppi_er', 'ppi_sbm'],
        ['power', 'power_conf', 'power_er', 'power_sbm'],
        ['wiki', 'wiki_conf', 'wiki_er', 'wiki_sbm']
    ]
    row_labels = ['Astro', 'PPI', 'Power', 'Wiki']
    col_labels = ['Base Network', 'Config Model', 'Erdos Renyi', 'Stoch. Block Model']
    
    # 4. Styles
    family_colors = {
        'ppi':   {'random': '#DA94A3', 'hub': '#E9204B', 'periphery': '#782235'}, 
        'astro': {'random': "#83E3B8", 'hub': "#089B3B", 'periphery': "#1F4836"}, 
        'power': {'random': "#BDA8F4", 'hub': '#5524E8', 'periphery': '#372278'}, 
        'wiki':  {'random': "#F4B266", 'hub': '#F6E825', 'periphery': "#925A04"}  
    }
    
    target_styles = {
        "random_target": {"ls": ":", "marker": "o", "label": "Random", "shade": "random"},
        "hub_target": {"ls": "-", "marker": "s", "label": "Hubs", "shade": "hub"},
        "periphery_target": {"ls": "--", "marker": "^", "label": "Periphery", "shade": "periphery"}
    }
    
    # 5. Figure Setup
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.15, wspace=0.1, top=0.92, bottom=0.1)

    algorithm_dict = {
        'diamond': 'Hypergeometric Signficance of Connectivity',
        'rwr_row': 'Random Walk with Restart (Row-Normalized)',
        'rwr_sym': 'Random Walk with Restart (Symmetric)',
        'first_neighbors': 'First Neighbors'}
    
    plt.suptitle(f"{algorithm_dict.get(algorithm)}, k={k_val}", fontsize=24, fontweight='bold')
    
    #tick_formatter = FuncFormatter(lambda val, pos: f"{val:g}%")
    tick_formatter = FuncFormatter(lambda val, pos: f"{int(val)}")
    
    full_x_ticks = [50, 60, 70, 80, 100, 125, 150, 200, 250, 300]
    if view == 'removal': custom_x_ticks = [t for t in full_x_ticks if t <= 100]
    elif view == 'addition': custom_x_ticks = [t for t in full_x_ticks if t >= 100]
    else: custom_x_ticks = full_x_ticks
    
    # 6. Plotting Loop
    for row_idx in range(4):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            net_id = grid_layout[row_idx][col_idx]
            
            fam = row_labels[row_idx].lower()
            palette = family_colors.get(fam, family_colors['power'])
            
            df_net = df_plot[df_plot['network'] == net_id]
            
            if not df_net.empty:
                # Local baseline anchor (noise_level == 0)
                baseline_row = df_net[df_net['noise_level'] == 0].copy()
                
                for t_name, style in target_styles.items():
                    df_pert = df_net[df_net['perturbation_type'] == t_name].copy()
                    
                    if not baseline_row.empty:
                        base_point = baseline_row.copy()
                        base_point['relative_size'] = 100.0
                        df_pert = pd.concat([df_pert, base_point])
                        
                    df_pert = df_pert.sort_values('relative_size')
                    
                    if view == 'removal': df_pert = df_pert[df_pert['relative_size'] <= 100]
                    elif view == 'addition': df_pert = df_pert[df_pert['relative_size'] >= 100]
                    
                    if not df_pert.empty:
                        x = df_pert['relative_size']
                        y = df_pert[mean_col]
                        std = df_pert[std_col]
                        c = palette[style['shade']]
                        
                        ax.plot(x, y, color=c, linestyle=style['ls'], marker=style['marker'], markersize=5, linewidth=2)
                        
                        # AUPRC, F1, Recall, Precision, Jaccard are all bounded [0, 1]
                        y_lower = np.maximum(0, y - std)
                        y_upper = np.minimum(1.0, y + std)
                        ax.fill_between(x, y_lower, y_upper, color=c, alpha=0.15)
            
            # Formatting
            ax.set_xscale('log')
            ax.set_xticks(custom_x_ticks)
            ax.set_xticks([], minor=True)
            ax.xaxis.set_major_formatter(tick_formatter)
            
            if view == 'removal': ax.set_xlim(right=100)
            elif view == 'addition': ax.set_xlim(left=100)
            
            ax.set_ylim(-0.05, 1.05)
            ax.axvline(100, color='gray', linestyle='-', alpha=0.3, linewidth=2)
            
            if row_idx == 0: ax.set_title(col_labels[col_idx], fontsize=16, fontweight='bold', pad=15)
            if col_idx == 0: ax.set_ylabel(f"{row_labels[row_idx]}\n{metric.upper()}", fontsize=14, fontweight='bold', labelpad=10)
            if row_idx == 3: ax.set_xlabel("Relative Network Size (%)", fontsize=12, labelpad=10)
            
            ax.grid(True, which="major", axis="both", color="#E0E0E0", linestyle="--", alpha=0.7)
            for spine in ax.spines.values(): spine.set_color('#CCCCCC')

    # 7. Legend
    legend_elements = [
        Line2D([0], [0], color='gray', lw=2.5, ls=style['ls'], marker=style['marker'], markersize=8, label=style['label'])
        for style in target_styles.values()
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False, fontsize=16, bbox_to_anchor=(0.5, 0.02))
    
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight')
    plt.show()

# =====================================================================
# Execution
# =====================================================================

# Load the dataframe
df = pd.read_csv("./outputs/seed_expansion/expansion/perturbed_summary_csvs/local_results_aggregated.csv")
for algorithm in ['diamond', 'rwr_row', 'rwr_sym', 'first_neighbors']:
    for k_val in ['10', '25', '50', '100', 'max']:
        for metric in ['auprc', 'f1', 'recall', 'precision', 'jaccard']:
            save_path = f"./outputs/seed_expansion/expansion/figures/{algorithm}_k{k_val}_{metric}.pdf"
            plot_local_robustness(df, algorithm=algorithm, k_val=k_val, metric=metric, save_fig=save_path)


# %%
