import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def load_and_merge_datasets(csv_paths: List[str]) -> pd.DataFrame:
    """Merges multiple experiment CSVs into a single Master DataFrame."""
    dfs = [pd.read_csv(path) for path in csv_paths]
    return pd.concat(dfs, ignore_index=True)


def plot_comparative_topology(
    csv_paths: List[str], save_fig: str = "topology_comparison.pdf"
):
    """
    Plots the topological breakdown (Modularity and GCC Fraction)
    comparing different network models (e.g., PPI vs ER vs SBM).
    """
    df = load_and_merge_datasets(csv_paths)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Modularity collapse under Edge Addition
    df_added = df[df["Noise_Type"] == "added"]
    if not df_added.empty:
        sns.lineplot(
            data=df_added,
            x="Noise_Level",
            y="Modularity_Q",
            hue="Network_Model",
            style="Network_Model",
            markers=True,
            dashes=False,
            ax=axes[0],
            errorbar="sd",
        )
        axes[0].set_title("Small-World Collapse (Added Edges)", weight="bold")
        axes[0].set_xlabel("Noise Level (Ratio of Added Edges)")
        axes[0].set_ylabel("Modularity (Q)")

    # Subplot 2: GCC shattering under Edge Deletion
    df_removed = df[df["Noise_Type"] == "removed"]
    if not df_removed.empty:
        sns.lineplot(
            data=df_removed,
            x="Noise_Level",
            y="GCC_Fraction",
            hue="Network_Model",
            style="Network_Model",
            markers=True,
            dashes=False,
            ax=axes[1],
            errorbar="sd",
        )
        axes[1].set_title("Percolation Threshold (Removed Edges)", weight="bold")
        axes[1].set_xlabel("Noise Level (Ratio of Removed Edges)")
        axes[1].set_ylabel("Fraction of Nodes in GCC")

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_fig}")


def plot_comparative_spectral_robustness(
    csv_paths: List[str], save_fig: str = "spectral_robustness.pdf"
):
    """
    Plots the stability of the Top 200 Hubs (Centrality) vs Communities (Fiedler).
    """
    df = load_and_merge_datasets(csv_paths)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df_added = df[df["Noise_Type"] == "added"]
    if not df_added.empty:
        sns.lineplot(
            data=df_added,
            x="Noise_Level",
            y="Overlap_Centrality",
            hue="Network_Model",
            style="Network_Model",
            markers=True,
            dashes=False,
            ax=axes[0],
            errorbar="sd",
        )
        axes[0].set_title("Hub Robustness to False Positives", weight="bold")
        axes[0].set_xlabel("Noise Level (Ratio of Added Edges)")
        axes[0].set_ylabel("Top 200 Centrality Overlap (Jaccard)")
        axes[0].set_ylim(-0.05, 1.05)

    df_removed = df[df["Noise_Type"] == "removed"]
    if not df_removed.empty:
        sns.lineplot(
            data=df_removed,
            x="Noise_Level",
            y="Overlap_Fiedler",
            hue="Network_Model",
            style="Network_Model",
            markers=True,
            dashes=False,
            ax=axes[1],
            errorbar="sd",
        )
        axes[1].set_title("Community Collapse to False Negatives", weight="bold")
        axes[1].set_xlabel("Noise Level (Ratio of Removed Edges)")
        axes[1].set_ylabel("Top 200 Fiedler Overlap (Jaccard)")
        axes[1].set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_fig}")
