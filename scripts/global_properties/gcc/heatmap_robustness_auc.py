from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_CSV = Path("outputs/summarizing_analysis/robustness_auc/robustness_auc.csv")
OUTPUT_DIR = Path("outputs/global_properties/gcc_singletons/figures/heatmap_rauc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INTEGRATION_RANGE = "0-0.5"  # Filter range (e.g. '0-0.5' or '0-1')

NETWORK_LABEL_MAP = {
    # PPI Block
    "ppi": "PPI",
    "ppi_conf": "Conf. Model (PPI)",
    "ppi_er": "ER Model (PPI)",
    "ppi_sbm": "SBM Model (PPI)",
    # Power Grid Block
    "power": "Power Grid",
    "power_conf": "Conf. Model (Power Grid)",
    "power_er": "ER Model (Power Grid)",
    "power_sbm": "SBM Model (Power Grid)",
    # AstroPh Block
    "astro": "AstroPh",
    "astro_conf": "Conf. Model (AstroPh)",
    "astro_er": "ER Model (AstroPh)",
    "astro_sbm": "SBM Model (AstroPh)",
    # Wiki-Vote Block
    "wiki": "Wiki-Vote",
    "wiki_conf": "Conf. Model (Wiki-Vote)",
    "wiki_er": "ER Model (Wiki-Vote)",
    "wiki_sbm": "SBM Model (Wiki-Vote)",
}

ROW_ORDER = [
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

TARGET_LABEL_MAP = {
    "random": "Random",
    "hub": "Hub",
    "periphery": "Periphery",
    "perturbed_random_target": "Random",
    "perturbed_hub_target": "Hub",
    "perturbed_periphery_target": "Periphery",
}

COLUMN_ORDER = ["Random", "Hub", "Periphery"]


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

    mask = (
        (df["action"] == "removal")
        & (df["algorithm"] == "gcc")
        & (df["integration_range"] == INTEGRATION_RANGE)
    )
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        print("⚠️ Warning: No rows matched the filter criteria!")
        return pd.DataFrame()

    filtered_df["network_clean"] = filtered_df["network"].map(NETWORK_LABEL_MAP)
    filtered_df["target_clean"] = filtered_df["target"].map(TARGET_LABEL_MAP)

    pivot_df = filtered_df.pivot_table(
        index="network_clean",
        columns="target_clean",
        values="auc_normalized",
        aggfunc="mean",
    )

    pivot_df = pivot_df.reindex(index=ROW_ORDER, columns=COLUMN_ORDER)
    return pivot_df


# ==============================================================================
# PLOTTING PIPELINE
# ==============================================================================
def plot_gcc_removal_heatmap_compact():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )

    if not INPUT_CSV.exists():
        print(f"❌ Input CSV not found at: {INPUT_CSV.resolve()}")
        return

    pivot_df = load_and_prepare_data(INPUT_CSV)

    if pivot_df.empty:
        return

    # Compact publication dimensions
    fig, ax = plt.subplots(figsize=(3.8, 4.8), dpi=300)

    cmap = sns.color_palette("vlag_r", as_cmap=True)

    sns.heatmap(
        pivot_df,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.5, "weight": "normal"},
        linewidths=0.5,  # Normal grid line width within blocks
        linecolor="#FFFFFF",
        square=False,
        cbar_kws={
            "label": r"GCC Robustness ($R_{\mathrm{AUC}}$)",
            "shrink": 0.7,
            "pad": 0.04,
        },
    )

    # Thicker white horizontal lines to mark block boundaries cleanly
    for y_boundary in [4, 8, 12]:
        ax.axhline(y_boundary, color="#FFFFFF", linewidth=3.5)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax.set_xlabel("Perturbation Target", labelpad=6, fontweight="bold")
    ax.set_ylabel("", labelpad=0)
    ax.tick_params(axis="both", which="both", length=0)

    plt.tight_layout()

    out_png = OUTPUT_DIR / "gcc_removal_heatmap_compact.png"
    out_pdf = OUTPUT_DIR / "gcc_removal_heatmap_compact.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(
        f"✅ Compact Heatmap saved to:\n   - {out_png.resolve()}\n   - {out_pdf.resolve()}"
    )
    plt.show()


if __name__ == "__main__":
    plot_gcc_removal_heatmap_compact()
