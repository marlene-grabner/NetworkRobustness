"""
Raw global-efficiency-vs-noise-level curves, one figure per network family
(base network + er/conf/sbm null models as separate subplots), mirroring
scripts/global_properties/algebraic_connectivity/analysis/plot_aggregated_results_per_network_group.py
for the Fiedler value.

Reads outputs/global_properties/global_efficiency/global_efficiency_baseline_and_perturbed.csv
(per-repeat rows), produced by aggregate_global_efficiency.py.
"""
import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

INPUT_PATH = "outputs/global_properties/global_efficiency/global_efficiency_baseline_and_perturbed.csv"


def load_aggregated():
    """Per-repeat raw CSV -> mean/std per condition, with noise_level converted
    to the 100 +/- N% relative-network-size scale used across the other plots."""
    df_raw = pd.read_csv(INPUT_PATH)

    df = (
        df_raw.groupby(["network", "noise_type", "action", "noise_level"])["global_efficiency"]
        .agg(avg_global_efficiency="mean", std_global_efficiency="std")
        .reset_index()
    )
    df["std_global_efficiency"] = df["std_global_efficiency"].fillna(0.0)

    def relative_size(row):
        if row["noise_type"] == "baseline":
            return 100.0
        level_pct = row["noise_level"] * 100
        return 100.0 - level_pct if row["action"] == "removal" else 100.0 + level_pct

    df["noise_level"] = df.apply(relative_size, axis=1)
    return df


# ============================================================================
# --- Plotting Function ---
# ===========================================================================

def plot_robustness_results(df, networks_to_plot, save_fig=None, view='all'):
    """
    Plots global efficiency vs. noise level for a specific list of networks.
    """
    df_filtered = df[df['network'].isin(networks_to_plot)].copy()

    if df_filtered.empty:
        print("None of the specified networks were found in the DataFrame.")
        return

    label_map = {
        'power': 'Western US Power Grid',
        'power_conf': 'Configuration Model (Power)',
        'power_er': 'Erdos Renyi (Power)',
        'power_sbm': 'Stochastic Block Model (Power)',
        'ppi': 'Protein Interaction Network',
        'ppi_conf': 'Configuration Model (PPI)',
        'ppi_er': 'Erdos Renyi (PPI)',
        'ppi_sbm': 'Stochastic Block Model (PPI)',
        'astro': 'Astrophysics Collaboration',
        'astro_conf': 'Configuration Model (Astro)',
        'astro_er': 'Erdos Renyi (Astro)',
        'astro_sbm': 'Stochastic Block Model (Astro)',
        'wiki': 'Wikipedia Vote',
        'wiki_conf': 'Configuration Model (Wiki)',
        'wiki_er': 'Erdos Renyi (Wiki)',
        'wiki_sbm': 'Stochastic Block Model (Wiki)'
    }

    def get_nice_title(net_id):
        return label_map.get(net_id, net_id.replace('_', ' ').title())

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
        "random": {"ls": ":", "marker": "o", "label": "Random", "shade": "random"},
        "hub": {"ls": "-", "marker": "s", "label": "Hubs", "shade": "hubs"},
        "periphery": {"ls": "--", "marker": "^", "label": "Periphery", "shade": "periphery"}
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
        df_net = df_filtered[df_filtered['network'] == net_id]

        if df_net.empty:
            ax.set_title(f"No data for: {net_id}")
            continue

        fam = get_family(net_id)
        palette = family_colors[fam]

        baseline_row = df_net[df_net['noise_type'] == 'baseline'].copy()

        for p_type, style in target_styles.items():
            df_pert = df_net[df_net['noise_type'] == p_type].copy()

            if not baseline_row.empty:
                base_point = baseline_row.copy()
                base_point['noise_level'] = 100.0
                df_pert = pd.concat([df_pert, base_point])

            df_pert = df_pert.sort_values('noise_level')

            if view == 'removal':
                df_pert = df_pert[df_pert['noise_level'] <= 100]
            elif view == 'addition':
                df_pert = df_pert[df_pert['noise_level'] >= 100]

            if df_pert.empty:
                continue

            x = df_pert['noise_level']
            y = df_pert['avg_global_efficiency']
            std = df_pert['std_global_efficiency']
            c = palette[style['shade']]

            ax.plot(x, y, color=c, linestyle=style['ls'], marker=style['marker'],
                    markersize=6, linewidth=2.5)
            ax.fill_between(x, y - std, y + std, color=c, alpha=0.15)

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
            base_val = baseline_row['avg_global_efficiency'].values[0]
            ax.axhline(base_val, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)

            x_min = ax.get_xlim()[0]
            ax.text(x_min, base_val + y_offset, ' Baseline Efficiency',
                    color='dimgray', va='bottom', ha='left', fontsize=12, style='italic')

        ax.axvline(100, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        if view == 'removal':
            ax.text(98, y_max, ' Base Graph (100%)', color='dimgray', va='top', ha='right', rotation=90, fontsize=12)
        else:
            ax.text(102, y_max, ' Base Graph (100%)', color='dimgray', va='top', ha='left', rotation=90, fontsize=12)

        ax.set_title(get_nice_title(net_id), fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Noise Level (%)", fontsize=14, labelpad=10)
        ax.set_ylabel("Global Efficiency", fontsize=14, labelpad=10)

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

    plt.close(fig)


# ============================================================================
# --- Reading in the dataframe ---
# ===========================================================================

df = load_aggregated()

# ============================================================================
# --- Plotting ---
# ===========================================================================
output_folder_figures = "outputs/global_properties/global_efficiency/figures"

plot_robustness_results(df, ["ppi", "ppi_er", "ppi_conf", "ppi_sbm"], save_fig=f"{output_folder_figures}/all_ppis_all_perturbations_global_efficiency.pdf")
plot_robustness_results(df, ["astro", "astro_er", "astro_conf", "astro_sbm"], save_fig=f"{output_folder_figures}/all_astros_all_perturbations_global_efficiency.pdf")
plot_robustness_results(df, ["power", "power_er", "power_conf", "power_sbm"], save_fig=f"{output_folder_figures}/all_powers_all_perturbations_global_efficiency.pdf")
plot_robustness_results(df, ["wiki", "wiki_er", "wiki_conf", "wiki_sbm"], save_fig=f"{output_folder_figures}/all_wikis_all_perturbations_global_efficiency.pdf")

plot_robustness_results(df, ["astro", "ppi", "power", "wiki"], save_fig=f"{output_folder_figures}/all_base_networks_all_perturbations_global_efficiency.pdf")

plot_robustness_results(df, ["ppi", "ppi_er", "ppi_conf", "ppi_sbm"], save_fig=f"{output_folder_figures}/all_ppis_removals_global_efficiency.pdf", view='removal')
plot_robustness_results(df, ["astro", "astro_er", "astro_conf", "astro_sbm"], save_fig=f"{output_folder_figures}/all_astros_removals_global_efficiency.pdf", view='removal')
plot_robustness_results(df, ["power", "power_er", "power_conf", "power_sbm"], save_fig=f"{output_folder_figures}/all_powers_removals_global_efficiency.pdf", view='removal')
plot_robustness_results(df, ["wiki", "wiki_er", "wiki_conf", "wiki_sbm"], save_fig=f"{output_folder_figures}/all_wikis_removals_global_efficiency.pdf", view='removal')

plot_robustness_results(df, ["ppi", "ppi_er", "ppi_conf", "ppi_sbm"], save_fig=f"{output_folder_figures}/all_ppis_additions_global_efficiency.pdf", view='addition')
plot_robustness_results(df, ["astro", "astro_er", "astro_conf", "astro_sbm"], save_fig=f"{output_folder_figures}/all_astros_additions_global_efficiency.pdf", view='addition')
plot_robustness_results(df, ["power", "power_er", "power_conf", "power_sbm"], save_fig=f"{output_folder_figures}/all_powers_additions_global_efficiency.pdf", view='addition')
plot_robustness_results(df, ["wiki", "wiki_er", "wiki_conf", "wiki_sbm"], save_fig=f"{output_folder_figures}/all_wikis_additions_global_efficiency.pdf", view='addition')
