from pathlib import Path
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# ==============================================================================
# CONFIGURATION & TOGGLES
# ==============================================================================
# Add any metric keys here that you want to drop from the figure:
EXCLUDE_METRICS = [
    "Nodes_Full",
    "Edges_Full",
    "N_Components",
    "N_Singletons",
    "N_NonGCC_Nodes",
    "GCC_Size",
    "GCC_Fraction",
    "Transitivity",
    "Metrics_Scope",
]

# File output settings
INPUT_CSV = Path("outputs/models/baseline_network_properties.csv")
OUTPUT_DIR = Path("outputs/models/figures/network_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean, scientific display labels for metric rows
METRIC_LABELS = {
    "Nodes": r"Nodes ($N$)",
    "Edges": r"Edges ($M$)",
    "Average_Degree": r"Avg. Degree ($\langle k \rangle$)",
    "Density": r"Density ($\rho$)",
    "GCC_Fraction": r"GCC Fraction",
    "Degree_Skew": r"Degree Skewness",
    "Degree_CV": r"Degree Coeff. of Var. ($CV_k$)",
    "Assortativity": r"Degree Assortativity ($r$)",
    "Transitivity": r"Global Clustering / Transitivity",
    "Clustering_Coefficient": r"Avg. Clustering Coeff. ($\bar{C}$)",
    "Modularity_Louvain": r"Modularity ($Q$)",
    "Algebraic_Connectivity": r"Algebraic Connectivity ($\lambda_2$)",
    "Spectral_Gap": r"Spectral Gap ($\Delta \lambda$)",
    "Avg_Path_Length": r"Avg. Path Length ($\langle d \rangle$)",
}

# Clean, capitalized network display labels
NETWORK_LABELS = {
    "Protein-Protein Interaction Network": "PPI",
    "Western US Power Grid Network": "Power Grid",
    "Astrophysics Collaboration Network": "AstroPh",
    "Wikipedia Vote Network": "Wiki-Vote",
}

# ==============================================================================
# DATA PARSING & FORMATTING
# ==============================================================================
df = pd.read_csv("outputs/models/baseline_network_properties.csv")


def format_raw_value(val: float, metric_key: str) -> str:
    """Formats raw numbers cleanly for cell text annotations."""
    if metric_key in ["Nodes", "Edges"]:
        return f"{int(val):,}"
    if abs(val) < 0.001 and val != 0:
        return f"{val:.1e}"
    if abs(val) < 0.01:
        return f"{val:.4f}"
    return f"{val:.2f}"


def load_and_preprocess():
    df = pd.read_csv(INPUT_CSV, index_col=0)

    # 1. Filter out excluded metrics
    df = df.drop(index=[m for m in EXCLUDE_METRICS if m in df.index])

    df = df.astype(float)

    # 2. Rename columns and rows for presentation
    df.columns = [NETWORK_LABELS.get(c, c) for c in df.columns]

    # 3. Create raw formatted text annotations matrix
    annot_df = pd.DataFrame(index=df.index, columns=df.columns)
    for metric in df.index:
        for net in df.columns:
            annot_df.loc[metric, net] = format_raw_value(df.loc[metric, net], metric)

    # 4. Compute Row-Wise Z-Scores
    # z = (x - mean) / std across the 4 networks
    means = df.mean(axis=1)
    stds = df.std(axis=1, ddof=0)  # Population std
    stds = stds.replace(0, 1)  # Prevent division by zero if std is 0

    z_df = df.sub(means, axis=0).div(stds, axis=0)

    # Update row index with clean display labels
    z_df.index = [METRIC_LABELS.get(m, m) for m in z_df.index]
    annot_df.index = z_df.index

    return z_df, annot_df


# ==============================================================================
# PLOTTING PIPELINE
# ==============================================================================
def plot_topology_heatmap():
    # Typography configuration
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "font.weight": "normal",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10.5,
            "figure.titlesize": 14,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
        }
    )

    z_df, annot_df = load_and_preprocess()

    num_rows = len(z_df)
    fig_height = max(5.0, num_rows * 0.45)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    # Symmetrical z-score range centered at 0
    max_z = max(1.5, float(np.abs(z_df.values).max()))

    # Heatmap rendering
    sns.heatmap(
        z_df,
        ax=ax,
        cmap="vlag",  # Scientific blue-white-red palette
        vmin=-max_z,
        vmax=max_z,
        center=0.0,
        annot=annot_df.values,  # Display raw values inside cells
        fmt="",
        annot_kws={"size": 10, "weight": "normal", "color": "#0F172A"},
        linewidths=0.8,
        linecolor="#FFFFFF",
        cbar_kws={"label": r"Row-Wise $Z$-Score", "shrink": 0.8},
    )

    ax.set_xlabel("Empirical Network", labelpad=10)
    ax.set_ylabel("Network Metric", labelpad=10)
    plt.xticks(rotation=0)

    plt.tight_layout()

    out_png = OUTPUT_DIR / "network_topology_zscore_heatmap.png"
    out_pdf = OUTPUT_DIR / "network_topology_zscore_heatmap.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(
        f"✅ Network Topology Heatmap saved to:\n   - {out_png.resolve()}\n   - {out_pdf.resolve()}"
    )
    plt.show()


if __name__ == "__main__":
    plot_topology_heatmap()
