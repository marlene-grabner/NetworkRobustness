"""
Figures for the repeat-sufficiency analysis. Run subsample_repeats.py first.

Writes to outputs/repeat_sufficiency/figures/:
  example_curve_bands.pdf, curve_sd_vs_repeats.pdf, agreement_vs_repeats.pdf

Run from the repository root:
  python scripts/repeat_sufficiency/plot_repeat_sufficiency.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from subsample_repeats import OUT_DIR, RETENTION, SUBSET_SIZES, retention_curves, subset_means

FIGURE_DIR = OUT_DIR + "figures/"

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"

FAMILIES = {
    "ari": ("ARI", "#2a78d6", "o"),
    "jaccard": ("Jaccard ($k$=50)", "#eb6834", "s"),
    "auprc": ("AUPRC ($k$=50)", "#1baf7a", "^"),
    "fiedler": ("Fiedler value", "#eda100", "D"),
    "global": ("Global efficiency", "#e87ba4", "v"),
    "gcc": ("GCC fraction", "#008300", "P"),
}
EXAMPLE = {"network": "power", "metric": "jaccard_diamond", "action": "removal", "target": "random"}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def size_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(SUBSET_SIZES)
    ax.set_xticklabels(SUBSET_SIZES)
    ax.minorticks_off()
    ax.set_xlabel("Repeats in the subset")


def plot_example():
    df = pd.read_parquet(OUT_DIR + "per_repeat_values.parquet")
    query = " and ".join(f"{k} == '{v}'" for k, v in EXAMPLE.items())
    curve = df.query(query)
    matrix = retention_curves(curve, EXAMPLE["metric"], None)
    levels = np.sort(curve["noise_level"].unique()) * 100
    full = np.nanmean(matrix, axis=0)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3), sharey=True)
    for ax, size in zip(axes, (5, 25)):
        means = subset_means(matrix, size, rng)
        low, high = np.percentile(means, [2.5, 97.5], axis=0)
        ax.fill_between(levels, low, high, color="#eb6834", alpha=0.3, linewidth=0,
                        label="95 % of subsets")
        ax.plot(levels, full, color="#eb6834", linewidth=2, marker="o", markersize=4,
                label="All 100 repeats")
        ax.axhline(RETENTION, color=TEXT_SECONDARY, linewidth=1, linestyle="--",
                   label="Failure threshold")
        ax.set_title(f"{size} repeats", fontsize=11, loc="left")
        ax.set_xlabel("Edges removed (%)")
    axes[0].set_ylabel("Jaccard ($k$=50)")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR + "example_curve_bands.pdf")
    fig.savefig(FIGURE_DIR + "example_curve_bands.png", dpi=130)


def plot_curve_sd(by_curve):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for family, (label, color, marker) in FAMILIES.items():
        values = by_curve[by_curve["family"] == family].groupby("size")["curve_sd"].median()
        ax.plot(values.index, values.values, color=color, marker=marker, linewidth=2,
                markersize=6, label=label)
    size_axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel("SD of the mean curve\n(fraction of baseline, median curve)")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR + "curve_sd_vs_repeats.pdf")
    fig.savefig(FIGURE_DIR + "curve_sd_vs_repeats.png", dpi=130)


def plot_agreement(by_curve, by_contrast):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)

    failing = by_curve[by_curve["fails_in_full"]]
    for family, (label, color, marker) in FAMILIES.items():
        values = failing[failing["family"] == family].groupby("size")["threshold_exact"].mean()
        axes[0].plot(values.index, values.values * 100, color=color, marker=marker,
                     linewidth=2, markersize=6, label=label)
    axes[0].set_title("Same failure step as all 100 repeats", fontsize=11, loc="left")
    axes[0].set_ylabel("Agreement (%)")
    axes[0].legend(loc="lower right", fontsize=8)

    styles = [
        ("sign_agreement", "Direction of difference", "#2a78d6", "o"),
        ("threshold_order_agreement", "Order of failure", "#eb6834", "s"),
    ]
    empirical_vs_null = by_contrast[by_contrast["kind"] == "empirical_vs_null"]
    for column, label, color, marker in styles:
        data = empirical_vs_null
        if column == "threshold_order_agreement":
            data = data[data["thresholds_differ_in_full"]]
        values = data.groupby("size")[column].mean()
        axes[1].plot(values.index, values.values * 100, color=color, marker=marker,
                     linewidth=2, markersize=6, label=label)
    axes[1].set_title("Empirical vs. null model", fontsize=11, loc="left")
    axes[1].legend(loc="lower right", fontsize=9)

    for ax in axes:
        size_axis(ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR + "agreement_vs_repeats.pdf")
    fig.savefig(FIGURE_DIR + "agreement_vs_repeats.png", dpi=130)


if __name__ == "__main__":
    os.makedirs(FIGURE_DIR, exist_ok=True)
    by_curve = pd.read_csv(OUT_DIR + "by_curve.csv")
    by_contrast = pd.read_csv(OUT_DIR + "by_contrast.csv")
    for df in (by_curve, by_contrast):
        df["family"] = df["metric"].str.split("_").str[0]

    plot_example()
    plot_curve_sd(by_curve)
    plot_agreement(by_curve, by_contrast)
