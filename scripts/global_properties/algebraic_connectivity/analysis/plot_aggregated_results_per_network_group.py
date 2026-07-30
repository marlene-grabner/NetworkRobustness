import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# ============================================================================
# --- Plotting Function ---
# ===========================================================================

def plot_robustness_results(df, networks_to_plot, save_fig=None, view='all'):
    """
    Plots the algebraic connectivity vs. noise level for a specific list of networks.
    """
    # 1. Filter the dataframe
    df_filtered = df[df['network_id'].isin(networks_to_plot)].copy()
    
    if df_filtered.empty:
        print("None of the specified networks were found in the DataFrame.")
        return
    
    label_map = {
        'power': 'Western US Power Grid',
        'power_config': 'Configuration Model (Power)',
        'power_er': 'Erdos Renyi (Power)',
        'power_sbm': 'Stochastic Block Model (Power)',
        'ppi': 'Protein Interaction Network',
        'ppi_config': 'Configuration Model (PPI)',
        'ppi_er': 'Erdos Renyi (PPI)',
        'ppi_sbm': 'Stochastic Block Model (PPI)',
        'astro': 'Astrophysics Collaboration',
        'astro_config': 'Configuration Model (Astro)',
        'astro_er': 'Erdos Renyi (Astro)',
        'astro_sbm': 'Stochastic Block Model (Astro)',
        'wiki': 'Wikipedia Vote',
        'wiki_config': 'Configuration Model (Wiki)',
        'wiki_er': 'Erdos Renyi (Wiki)',
        'wiki_sbm': 'Stochastic Block Model (Wiki)'
    }
    
    def get_nice_title(net_id):
        if net_id in label_map:
            return label_map[net_id]
        for key, val in label_map.items():
            if net_id in key:
                return val
        return net_id.replace('_', ' ').title()

    family_colors = {
        'ppi':   {'random': '#DA94A3', 'hubs': '#E9204B', 'periphery': '#782235'}, 
        'astro': {'random': "#83E3B8", 'hubs': "#089B3B", 'periphery': "#1F4836"}, 
        'power': {'random': "#BDA8F4", 'hubs': '#5524E8', 'periphery': '#372278'}, 
        'wiki':  {'random': "#F4B266", 'hubs': '#F6E825', 'periphery': "#925A04"}  
    }
    
    def get_family(net_id):
        net_id_lower = net_id.lower()
        if 'ppi' in net_id_lower: return 'ppi'
        if 'astro' in net_id_lower: return 'astro'
        if 'wiki' in net_id_lower: return 'wiki'
        return 'power'
    
    families_in_plot = {get_family(net_id) for net_id in networks_to_plot}
    if len(families_in_plot) == 1:
        legend_palette = family_colors[list(families_in_plot)[0]]
    else:
        legend_palette = {'random': 'gray', 'hubs': 'gray', 'periphery': 'gray'}

    target_styles = {
        "perturbed_random_target": {"ls": ":", "marker": "o", "label": "Random", "shade": "random"},
        "perturbed_hub_target": {"ls": "-", "marker": "s", "label": "Hubs", "shade": "hubs"},
        "perturbed_periphery_target": {"ls": "--", "marker": "^", "label": "Periphery", "shade": "periphery"}
    }
    
    n_plots = len(networks_to_plot)
    cols = 2 if n_plots > 1 else 1
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 6 * rows), squeeze=False)
    fig.subplots_adjust(bottom=0.18 if rows == 1 else 0.12, hspace=0.4, wspace=0.2)
    axes = axes.flatten() 
    
    tick_formatter = FuncFormatter(lambda val, pos: f"{val:g}%")
    
    full_x_ticks = [50, 60, 70, 80, 100, 125, 150, 175, 200, 300]
    if view == 'removal':
        custom_x_ticks = [t for t in full_x_ticks if t <= 100]
    elif view == 'addition':
        custom_x_ticks = [t for t in full_x_ticks if t >= 100]
    else:
        custom_x_ticks = full_x_ticks
    
    for i, net_id in enumerate(networks_to_plot):
        ax = axes[i]
        df_net = df_filtered[df_filtered['network_id'] == net_id]
        
        if df_net.empty:
            ax.set_title(f"No data for: {net_id}")
            continue
            
        fam = get_family(net_id)
        palette = family_colors[fam]
        
        # Grab the baseline row once for this network
        baseline_row = df_net[df_net['noise_type'] == 'baseline'].copy()
        
        for p_type, style in target_styles.items():
            df_pert = df_net[df_net['noise_type'] == p_type].copy()
            
            # --- NEW FIX: Inject the baseline point so the line anchors at 100% ---
            if not baseline_row.empty:
                base_point = baseline_row.copy()
                base_point['noise_level'] = 100.0 # Force it to align with the 100% line
                df_pert = pd.concat([df_pert, base_point])
                
            # Sort so the line draws correctly from left to right
            df_pert = df_pert.sort_values('noise_level')
            
            # Filter the data points based on the view
            if view == 'removal':
                df_pert = df_pert[df_pert['noise_level'] <= 100]
            elif view == 'addition':
                df_pert = df_pert[df_pert['noise_level'] >= 100]
                
            if df_pert.empty:
                continue
                
            x = df_pert['noise_level']
            y = df_pert['avg_algebraic_connectivity']
            std = df_pert['std_algebraic_connectivity']
            c = palette[style['shade']]
            
            ax.plot(x, y, color=c, linestyle=style['ls'], marker=style['marker'], 
                    markersize=6, linewidth=2.5)
            ax.fill_between(x, y - std, y + std, color=c, alpha=0.15)
            
        # --- Axis Scaling and Tick Formatting ---
        ax.set_xscale('log')
        
        ax.set_xticks(custom_x_ticks)
        ax.set_xticks([], minor=True) 
        ax.xaxis.set_major_formatter(tick_formatter)
        
        if view == 'removal':
            ax.set_xlim(right=100)
        elif view == 'addition':
            ax.set_xlim(left=100)
            
        ax.tick_params(axis='x', which='major', labelsize=13, rotation=45)
        ax.tick_params(axis='y', which='major', labelsize=13)
        
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        y_offset = y_range * 0.03 
        
        if not baseline_row.empty:
            base_val = baseline_row['avg_algebraic_connectivity'].values[0]
            ax.axhline(base_val, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
            
            x_min = ax.get_xlim()[0]
            ax.text(x_min, base_val + y_offset, ' Baseline Connectivity', 
                    color='dimgray', va='bottom', ha='left', fontsize=12, style='italic')
        
        ax.axvline(100, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        if view == 'removal':
            ax.text(98, y_max, ' Base Graph (100%)', color='dimgray', va='top', ha='right', rotation=90, fontsize=12)
        else:
            ax.text(102, y_max, ' Base Graph (100%)', color='dimgray', va='top', ha='left', rotation=90, fontsize=12)
        
        ax.set_title(get_nice_title(net_id), fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Noise Level (%)", fontsize=14, labelpad=10)
        ax.set_ylabel("Algebraic Connectivity", fontsize=14, labelpad=10)
        
        ax.grid(True, which="major", axis="both", color="#E0E0E0", linestyle="-", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    legend_elements = [
        Line2D([0], [0], color=legend_palette[style['shade']], lw=2.5, 
               ls=style['ls'], marker=style['marker'], markersize=8, label=style['label'])
        for style in target_styles.values()
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False, 
               fontsize=14, bbox_to_anchor=(0.5, 0.01))
    
    if save_fig:
        plt.savefig(save_fig, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_fig}")
        
    plt.show()

# ============================================================================
# --- Reading in the dataframe ---
# ===========================================================================

df = pd.read_csv("outputs/global_properties/algebraic_connectivity/aggregated_mean_fiedler_values_perturbed_plus_baseline.csv")

# ============================================================================
# --- Plotting ---
# ===========================================================================
output_folder_figures = "outputs/global_properties/algebraic_connectivity/figures"

plot_robustness_results(df, ["ppi", "ppi_er", "ppi_conf", "ppi_sbm"], save_fig=f"{output_folder_figures}/all_ppis_all_perturbations_fiedler_value.pdf")
plot_robustness_results(df, ["astro", "astro_er", "astro_conf", "astro_sbm"], save_fig=f"{output_folder_figures}/all_astros_all_perturbations_fiedler_value.pdf")
plot_robustness_results(df, ["power", "power_er", "power_conf", "power_sbm"], save_fig=f"{output_folder_figures}/all_powers_all_perturbations_fiedler_value.pdf")
plot_robustness_results(df, ["wiki", "wiki_er", "wiki_conf", "wiki_sbm"], save_fig=f"{output_folder_figures}/all_wikis_all_perturbations_fiedler_value.pdf")

plot_robustness_results(df, ["astro", "ppi", "power", "wiki"], save_fig=f"{output_folder_figures}/all_base_networks_all_perturbations_fiedler_value.pdf")

plot_robustness_results(df, ["ppi", "ppi_er", "ppi_conf", "ppi_sbm"], save_fig=f"{output_folder_figures}/all_ppis_removals_fiedler_value.pdf", view='removal')
plot_robustness_results(df, ["astro", "astro_er", "astro_conf", "astro_sbm"], save_fig=f"{output_folder_figures}/all_astros_removals_fiedler_value.pdf", view='removal')
plot_robustness_results(df, ["power", "power_er", "power_conf", "power_sbm"], save_fig=f"{output_folder_figures}/all_powers_removals_fiedler_value.pdf", view='removal')
plot_robustness_results(df, ["wiki", "wiki_er", "wiki_conf", "wiki_sbm"], save_fig=f"{output_folder_figures}/all_wikis_removals_fiedler_value.pdf", view='removal')

plot_robustness_results(df, ["ppi", "ppi_er", "ppi_conf", "ppi_sbm"], save_fig=f"{output_folder_figures}/all_ppis_additions_fiedler_value.pdf", view='addition')
plot_robustness_results(df, ["astro", "astro_er", "astro_conf", "astro_sbm"], save_fig=f"{output_folder_figures}/all_astros_additions_fiedler_value.pdf", view='addition')
plot_robustness_results(df, ["power", "power_er", "power_conf", "power_sbm"], save_fig=f"{output_folder_figures}/all_powers_additions_fiedler_value.pdf", view='addition')
plot_robustness_results(df, ["wiki", "wiki_er", "wiki_conf", "wiki_sbm"], save_fig=f"{output_folder_figures}/all_wikis_additions_fiedler_value.pdf", view='addition')