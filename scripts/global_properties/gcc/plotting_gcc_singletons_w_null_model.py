import os, re, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


####################################################################################
# Parameters
input_df_path = "./outputs/global_properties/aggregated_gcc_singletons.csv"
output_figures_path = "./outputs/global_properties/figures/gcc_singletons/"
####################################################################################

######################
#   Functions
######################

# ===================================================
# Plotting function for GCC
# ===================================================


def plot_gcc_robustness(
    df_agg,
    networks,
    color_dict,
    measure="gcc",
    metric="mean",
    noise_type_filter=None,
    save_fig=None,
):
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
        "western_us_power_grid": "Western US Power Grid",
        "western_us_power_grid_config": "Configuration Model (Power)",
        "western_us_power_grid_er": "Erdos Renyi (Power)",
        "western_us_power_grid_sbm": "Stochastic Block Model (Power)",
        "chloe_ppi_lcc_2026_02_23": "Protein Interaction Network",
        "chloe_ppi_lcc_2026_02_23_config": "Configuration Model (PPI)",
        "chloe_ppi_lcc_2026_02_23_er": "Erdos Renyi (PPI)",
        "chloe_ppi_lcc_2026_02_23_sbm": "Stochastic Block Model (PPI)",
        "ca-AstroPH_gcc": "Astrophysics Collaboration",
        "ca-AstroPH_gcc_config": "Configuration Model (Astro)",
        "ca-AstroPH_gcc_er": "Erdos Renyi (Astro)",
        "ca-AstroPH_gcc_sbm": "Stochastic Block Model (Astro)",
        "wiki-Vote_gcc": "Wikipedia Vote",
        "wiki-Vote_gcc_config": "Configuration Model (Wiki)",
        "wiki-Vote_gcc_er": "Erdos Renyi (Wiki)",
        "wiki-Vote_gcc_sbm": "Stochastic Block Model (Wiki)",
    }

    # 1. Base Filter: Only removals, and only the networks we asked for
    plot_df = df_agg[
        (df_agg["action"] == "removal") & (df_agg["network"].isin(networks))
    ].copy()

    if noise_type_filter:
        plot_df = plot_df[plot_df["noise_type"] == noise_type_filter]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # 2. Iterate and Plot
    for network in networks:
        net_data = plot_df[plot_df["network"] == network].sort_values(by="noise_level")

        if net_data.empty:
            continue

        x = net_data["noise_level"].values

        # 3. Determine measure and metric logic
        if measure == "gcc":
            if metric == "mean":
                y = net_data["gcc_mean"].values
                spread = net_data["gcc_std"].values
                lower_bound = y - spread
                upper_bound = y + spread
            elif metric == "median":
                y = net_data["gcc_median"].values
                half_iqr = net_data["gcc_iqr"].values / 2.0
                lower_bound = y - half_iqr
                upper_bound = y + half_iqr
            ylabel = "GCC Size (Mean ± Std)"
        elif measure == "num_singletons":
            if metric == "mean":
                y = net_data["num_singletons_mean"].values
                spread = net_data["num_singletons_std"].values
                lower_bound = y - spread
                upper_bound = y + spread
            elif metric == "median":
                y = net_data["num_singletons_median"].values
                half_iqr = net_data["num_singletons_iqr"].values / 2.0
                lower_bound = y - half_iqr
                upper_bound = y + half_iqr
            ylabel = "Number of Singletons (Mean ± Std)"

        else:
            raise ValueError(
                "Measure must be either 'gcc' or 'num_singletons', and metric must be either 'mean' or 'median'"
            )

        lower_bound = np.clip(lower_bound, 0.0, 1.0)
        upper_bound = np.clip(upper_bound, 0.0, 1.0)

        # 4. Draw the elements
        color = color_dict.get(network, "black")

        # Get the pretty name for the legend, fallback to raw name if missing
        legend_label = label_map.get(network, network)

        ax.plot(
            x, y, color=color, label=legend_label, linewidth=2, marker="o", markersize=5
        )
        ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2)

    # 5. Formatting (Larger fonts, no bolding, no title)
    ax.set_xlabel("Noise Level (Fraction Removed)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # Larger tick fonts
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Position legend at the top, spread into columns
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fontsize=12,
    )

    # Calculate tight bounding box so legend isn't clipped
    plt.tight_layout()

    # 6. Save option
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")


######################
#   Analysis - Plotting
######################

df = pd.read_csv(input_df_path)


network_colors = {
    "western_us_power_grid": "#372278",
    "western_us_power_grid_config": "#5524E8",
    "western_us_power_grid_er": "#6A6085",
    "western_us_power_grid_sbm": "#C8BEE3",
    "chloe_ppi_lcc_2026_02_23": "#782235",
    "chloe_ppi_lcc_2026_02_23_config": "#DA94A3",
    "chloe_ppi_lcc_2026_02_23_er": "#E9204B",
    "chloe_ppi_lcc_2026_02_23_sbm": "#1C0006",
    "ca-AstroPH_gcc": "#227851",
    "ca-AstroPH_gcc_config": "#BDF0D9",
    "ca-AstroPH_gcc_er": "#16C553",
    "ca-AstroPH_gcc_sbm": "#486055",
    "wiki-Vote_gcc": "#E8AD0C",
    "wiki-Vote_gcc_config": "#474501",
    "wiki-Vote_gcc_er": "#F6E825",
    "wiki-Vote_gcc_sbm": "#F1DDA6",
}

power_networks = [
    "western_us_power_grid",
    "western_us_power_grid_config",
    "western_us_power_grid_er",
    "western_us_power_grid_sbm",
]
ppi_networks = [
    "chloe_ppi_lcc_2026_02_23",
    "chloe_ppi_lcc_2026_02_23_config",
    "chloe_ppi_lcc_2026_02_23_er",
    "chloe_ppi_lcc_2026_02_23_sbm",
]
collab_networks = [
    "ca-AstroPH_gcc",
    "ca-AstroPH_gcc_config",
    "ca-AstroPH_gcc_er",
    "ca-AstroPH_gcc_sbm",
]
wiki_networks = [
    "wiki-Vote_gcc",
    "wiki-Vote_gcc_config",
    "wiki-Vote_gcc_er",
    "wiki-Vote_gcc_sbm",
]

all_network_sets = [power_networks, ppi_networks, collab_networks, wiki_networks]
perturbation_types = [
    "perturbed_periphery_target",
    "perturbed_hub_target",
    "perturbed_random_target",
]

for network in all_network_sets:
    for perturbation in perturbation_types:
        for measure in ["gcc", "num_singletons"]:
            for metric in ["mean", "median"]:
                plot_gcc_robustness(
                    df_agg=df,
                    networks=network,
                    color_dict=network_colors,
                    measure=measure,
                    metric=metric,  # Toggle to 'mean' if preferred
                    noise_type_filter=perturbation,  # Optional
                    save_fig=f"{output_figures_path}/{network[0]}_{perturbation}_average_{measure}_{metric}.pdf",
                )
