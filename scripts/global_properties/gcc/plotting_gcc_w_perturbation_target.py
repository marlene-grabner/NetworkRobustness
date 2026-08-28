import colorsys
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


####################################################################################
# Parameters
input_df_path = "./outputs/global_properties/aggregated_gcc_singletons.csv"
output_figures_path = (
    "./outputs/global_properties/gcc_singletons/figures/gcc_all_targets/"
)
####################################################################################

######################
#   Functions
######################

# ===================================================
# Plotting function for GCC
# ===================================================


def adjust_lightness(hex_color, amount=0.0):
    """Adjusts the lightness of a hex color.

    amount < 0 makes it darker, amount > 0 makes it lighter.
    """
    rgb = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    new_l = max(0.0, min(1.0, l + amount))
    new_rgb = colorsys.hls_to_rgb(h, new_l, s)
    return mcolors.to_hex(new_rgb)


def get_network_target_colors(network_str):
    """Returns curated shades (Dark Hub, Base Random, Light Periphery) matching

    the paper's network color scheme.
    """
    s = str(network_str).lower()

    if any(k in s for k in ["astro", "ca-astroph"]):
        # Green Family
        return {"hub": "#1b5e20", "random": "#2ea44f", "periphery": "#81c784"}

    elif any(k in s for k in ["wiki"]):
        # Yellow / Gold Family
        return {"hub": "#b45309", "random": "#eab308", "periphery": "#fde047"}

    elif any(k in s for k in ["ppi", "chloe"]):
        # Red / Rose Family
        return {"hub": "#881337", "random": "#dc2626", "periphery": "#fca5a5"}

    elif any(k in s for k in ["power", "western"]):
        # Blue Family
        return {"hub": "#1e3a8a", "random": "#2563eb", "periphery": "#93c5fd"}

    else:
        # Fallback Palette
        return {"random": "#334155", "hub": "#D64550", "periphery": "#2A9D8F"}


def plot_gcc_targets(
    df_agg,
    network,
    measure="gcc",
    metric="mean",
    base_color=None,
    target_colors=None,
    target_markers=None,
    save_fig=None,
):
    """Plots GCC or singletons robustness curves for a single network across 3 perturbation targets.

    Parameters:
    * df_agg: Aggregated pandas DataFrame.
    * network: String specifying which network to plot (e.g., 'ca-AstroPH_gcc').
    * measure: 'gcc' or 'num_singletons'.
    * metric: 'mean' or 'median'.
    * base_color: Optional single hex string. If provided, automatically generates dark/light shades.
    * target_colors: Dict mapping target keys to colors.
    * target_markers: Dict mapping target keys to matplotlib markers.
    * save_fig: Optional string path to save figure (e.g. 'gcc_targets.pdf').
    """

    # 1. Determine Target Colors
    if target_colors is None:
        if base_color is not None:
            # Generate lighter/darker shades dynamically from a single base color
            target_colors = {
                "hub": adjust_lightness(base_color, amount=-0.15),  # Darker
                "random": base_color,  # Base
                "periphery": adjust_lightness(base_color, amount=+0.20),  # Lighter
            }
        else:
            # Auto-select preset domain palette based on network name
            target_colors = get_network_target_colors(network)

    if target_markers is None:
        target_markers = {"random": "o", "hub": "s", "periphery": "^"}

    target_label_map = {
        "random": "Random Target",
        "hub": "Hub Target",
        "periphery": "Periphery Target",
    }

    # 2. Filter data
    plot_df = df_agg[
        (df_agg["action"] == "removal") & (df_agg["network"] == network)
    ].copy()

    if plot_df.empty:
        print(f"⚠️ No data found for network: {network}")
        return

    plot_df["target_clean"] = (
        plot_df["noise_type"]
        .astype(str)
        .str.replace("perturbed_", "", regex=False)
        .str.replace("_target", "", regex=False)
        .str.lower()
    )

    targets = ["random", "hub", "periphery"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Iterate over targets and plot
    for tgt in targets:
        tgt_data = plot_df[plot_df["target_clean"] == tgt].sort_values(by="noise_level")

        if tgt_data.empty:
            continue

        x = tgt_data["noise_level"].values

        if measure == "gcc":
            if metric == "mean":
                y = tgt_data["gcc_mean"].values
                spread = tgt_data["gcc_std"].values
                lower_bound = y - spread
                upper_bound = y + spread
            elif metric == "median":
                y = tgt_data["gcc_median"].values
                half_iqr = tgt_data["gcc_iqr"].values / 2.0
                lower_bound = y - half_iqr
                upper_bound = y + half_iqr
            ylabel = "GCC Size (Mean ± Std)"
            lower_bound = np.clip(lower_bound, 0.0, 1.0)
            upper_bound = np.clip(upper_bound, 0.0, 1.0)

        elif measure == "num_singletons":
            if metric == "mean":
                y = tgt_data["num_singletons_mean"].values
                spread = tgt_data["num_singletons_std"].values
                lower_bound = y - spread
                upper_bound = y + spread
            elif metric == "median":
                y = tgt_data["num_singletons_median"].values
                half_iqr = tgt_data["num_singletons_iqr"].values / 2.0
                lower_bound = y - half_iqr
                upper_bound = y + half_iqr
            ylabel = "Number of Singletons (Mean ± Std)"
            lower_bound = np.maximum(lower_bound, 0.0)

        else:
            raise ValueError("Measure must be either 'gcc' or 'num_singletons'")

        color = target_colors.get(tgt, "#333333")
        marker = target_markers.get(tgt, "o")
        legend_label = target_label_map.get(tgt, tgt.title())

        ax.plot(
            x,
            y,
            color=color,
            label=legend_label,
            linewidth=2,
            marker=marker,
            markersize=6,
        )
        ax.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2)

    # 4. Styling & Aesthetics
    ax.set_xlabel("Noise Level (Fraction Removed)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        fontsize=12,
    )

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")

    plt.show()


######################
#   Analysis - Plotting
######################

df = pd.read_csv(input_df_path)

empirical_networks = {
    "chloe_ppi_lcc_2026_02_23": "#bf2323",
    "western_us_power_grid": "#20118E",
    "ca-AstroPH_gcc": "#2a8f31",
    "wiki-Vote_gcc": "#f1de4f",
}

for network in empirical_networks:
    plot_gcc_targets(
        df_agg=df,
        network=network,
        measure="gcc",
        metric="mean",
        save_fig=f"{output_figures_path}/{network}_gcc_all_targets.pdf",
    )
