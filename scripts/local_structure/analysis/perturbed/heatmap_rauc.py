from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==============================================================================
# CONFIGURATION & TOGGLES
# ==============================================================================
# Set True for the 4 empirical networks only; False for all 16 network models
EMPIRICAL_ONLY = True

INTEGRATION_RANGE = "0-0.5"

INPUT_CSV = Path("outputs/summarizing_analysis/robustness_auc/robustness_auc.csv")
OUTPUT_DIR = Path("outputs/local_structure/figures/perturbed/rauc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# Mappings & Ordering
# ------------------------------------------------------------------------------
ALGO_MAP = {
    "infomap": "Infomap",
    "louvain": "Louvain",
    "leiden": "Leiden",
    "label_propagation": "Label Propagation",
    "labelprop": "Label Propagation",
}

ALGO_ORDER = ["Infomap", "Louvain", "Leiden", "Label Propagation"]

# Exact 6-column layout (3 Addition + 3 Removal)
COMBO_COL_ORDER = [
    "addition_random",
    "addition_hub",
    "addition_periphery",
    "removal_random",
    "removal_hub",
    "removal_periphery",
]

COL_DISPLAY_LABELS = ["Rand.", "Hub", "Perip.", "Rand.", "Hub", "Perip."]

# Network display labels
NETWORK_LABEL_MAP = {
    # Empirical
    "ppi": "PPI",
    "power": "Power Grid",
    "astro": "AstroPh",
    "wiki": "Wiki-Vote",
    # PPI Nulls
    "ppi_conf": "Conf. Model (PPI)",
    "ppi_er": "ER Model (PPI)",
    "ppi_sbm": "SBM Model (PPI)",
    # Power Grid Nulls
    "power_conf": "Conf. Model (Power Grid)",
    "power_er": "ER Model (Power Grid)",
    "power_sbm": "SBM Model (Power Grid)",
    # AstroPh Nulls
    "astro_conf": "Conf. Model (AstroPh)",
    "astro_er": "ER Model (AstroPh)",
    "astro_sbm": "SBM Model (AstroPh)",
    # Wiki-Vote Nulls
    "wiki_conf": "Conf. Model (Wiki-Vote)",
    "wiki_er": "ER Model (Wiki-Vote)",
    "wiki_sbm": "SBM Model (Wiki-Vote)",
}

EMPIRICAL_ROW_ORDER = ["PPI", "Power Grid", "AstroPh", "Wiki-Vote"]

FULL_16_ROW_ORDER = [
    "PPI",
    "Conf. Model (PPI)",
    "ER Model (PPI)",
    "SBM Model (PPI)",
    "Power Grid",
    "Conf. Model (Power Grid)",
    "ER Model (Power Grid)",
    "SBM Model (Power Grid)",
    "AstroPh",
    "Conf. Model (AstroPh)",
    "ER Model (AstroPh)",
    "SBM Model (AstroPh)",
    "Wiki-Vote",
    "Conf. Model (Wiki-Vote)",
    "ER Model (Wiki-Vote)",
    "SBM Model (Wiki-Vote)",
]

ROW_ORDER = EMPIRICAL_ROW_ORDER if EMPIRICAL_ONLY else FULL_16_ROW_ORDER


# ==============================================================================
# DATA LOADING & PARSING
# ==============================================================================
def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["action"] = df["action"].astype(str).str.lower().str.strip()
    df["algorithm"] = df["algorithm"].astype(str).str.lower().str.strip()
    df["network"] = df["network"].astype(str).str.lower().str.strip()
    df["target"] = df["target"].astype(str).str.lower().str.strip()
    df["integration_range"] = df["integration_range"].astype(str).str.strip()

    df["auc_normalized"] = pd.to_numeric(df["auc_normalized"], errors="coerce")

    df = df[df["integration_range"] == INTEGRATION_RANGE].copy()

    df["algo_clean"] = df["algorithm"].map(ALGO_MAP)
    df = df[df["algo_clean"].isin(ALGO_ORDER)].copy()

    df["net_clean"] = df["network"].map(NETWORK_LABEL_MAP)
    df = df[df["net_clean"].isin(ROW_ORDER)].copy()

    df["combo_key"] = df["action"] + "_" + df["target"]

    return df


# ==============================================================================
# PLOTTING PIPELINE
# ==============================================================================
def plot_mesoscale_community_heatmaps_compact():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8 if EMPIRICAL_ONLY else 7,
        }
    )

    if not INPUT_CSV.exists():
        print(f"❌ Input CSV not found at: {INPUT_CSV.resolve()}")
        return

    df_plot = load_and_prepare_data(INPUT_CSV)

    if df_plot.empty:
        print("⚠️ No matching data found.")
        return

    # Compact Figure Dimensions
    fig_height = 2.8 if EMPIRICAL_ONLY else 5.0
    fig, axes = plt.subplots(1, 4, figsize=(8.5, fig_height), sharey=True, dpi=300)

    cmap = sns.color_palette("vlag_r", as_cmap=True)

    for idx, algo_name in enumerate(ALGO_ORDER):
        ax = axes[idx]
        sub = df_plot[df_plot["algo_clean"] == algo_name]

        pivot_df = sub.pivot_table(
            index="net_clean",
            columns="combo_key",
            values="auc_normalized",
            aggfunc="mean",
        )

        # Reindex rows and columns strictly
        pivot_df = pivot_df.reindex(index=ROW_ORDER, columns=COMBO_COL_ORDER)

        # Render Heatmap
        sns.heatmap(
            pivot_df,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            annot_kws={
                "size": 6.8 if EMPIRICAL_ONLY else 5.8,
                "weight": "normal",
            },
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="#FFFFFF",
            cbar=False,
            square=False,
        )

        # 1. Subtle vertical gap between Addition (cols 0-2) and Removal (cols 3-5)
        ax.axvline(3, color="#FFFFFF", linewidth=3.5)

        # 2. Horizontal block boundary lines if showing 16 network models
        if not EMPIRICAL_ONLY:
            for y_boundary in [4, 8, 12]:
                ax.axhline(y_boundary, color="#FFFFFF", linewidth=2.5)

        # Panel Title
        ax.set_title(algo_name, pad=18, fontweight="bold")

        # X-Axis Ticks & Headers
        ax.set_xticks(np.arange(len(COL_DISPLAY_LABELS)) + 0.5)
        ax.set_xticklabels(COL_DISPLAY_LABELS, rotation=45, ha="center")
        ax.set_xlabel("")

        # Top Section Banners ("Addition" & "Removal")
        y_loc = -0.2 if EMPIRICAL_ONLY else -0.35
        ax.text(
            1.5,
            y_loc,
            "Addition",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#334155",
        )
        ax.text(
            4.5,
            y_loc,
            "Removal",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#334155",
        )

        # Y-ticks on leftmost panel only
        if idx == 0:
            ax.set_ylabel("", labelpad=6)
            ax.tick_params(axis="y", labelleft=True, left=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, left=False)

        ax.tick_params(axis="x", length=0)

    # --------------------------------------------------------------------------
    # BOTTOM HORIZONTAL COLORBAR
    # --------------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0.12, 1, 0.94])

    cbar_y_pos = 0.1 if EMPIRICAL_ONLY else 0.09
    cbar_x_start = 0.24 if EMPIRICAL_ONLY else 0.275
    cbar_ax = fig.add_axes([cbar_x_start, cbar_y_pos, 0.6, 0.03])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"Partition Robustness ($R_{\mathrm{AUC}}$)", fontsize=8, labelpad=4)
    cbar.ax.tick_params(labelsize=7.5)

    # --------------------------------------------------------------------------
    # SAVE EXPORTS
    # --------------------------------------------------------------------------
    out_suffix = f"{'empirical' if EMPIRICAL_ONLY else '16models'}"
    out_png = (
        OUTPUT_DIR
        / f"mesoscale_algorithms_rauc_summary_{out_suffix}_{INTEGRATION_RANGE.replace('.', 'p')}.png"
    )
    out_pdf = (
        OUTPUT_DIR
        / f"mesoscale_algorithms_rauc_summary_{out_suffix}_{INTEGRATION_RANGE.replace('.', 'p')}.pdf"
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(
        f"✅ R-AUC Summary Heatmap ({out_suffix}) saved to:\n   - {out_png.resolve()}\n   - {out_pdf.resolve()}"
    )


if __name__ == "__main__":
    plot_mesoscale_community_heatmaps_compact()
