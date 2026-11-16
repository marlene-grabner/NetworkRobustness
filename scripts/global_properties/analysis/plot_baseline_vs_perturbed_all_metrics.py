"""
Baseline vs perturbed for all global properties computed by
scripts/global_properties/calculate_global_properties.py.

Reads
  outputs/global_properties/all_metrics/{network}/{model}/{noise_type}_{level}.parquet
  outputs/global_properties/all_metrics/{network}/{model}/baseline.csv

Change vs baseline (per perturbed network, against the baseline of the same
network/model/replicate):
  - relative change (p - b) / b  for unbounded positive metrics
    (efficiency, transitivity, max_core, largest_eigenvalue)
  - difference p - b             for fractions / signed metrics
    (gcc, max_core_frac, assortativity) and num_singletons
  - the value itself             for the similarity metrics (rank correlation
    and top-k overlap with the baseline); 1 = identical to the baseline
The heatmaps and null-model figures show the relative metrics as
log2(perturbed / baseline) instead, which stays readable when the baseline is
close to zero (e.g. transitivity of ER null models) and treats a halving and a
doubling symmetrically. num_singletons is a raw count, so it is left out of the
figures that compare networks (the GCC fraction covers fragmentation there).

Writes (outputs/global_properties/figures/baseline_vs_perturbed/)
  summary/summary_empirical_networks.pdf
      median change across all empirical networks vs noise level (IQR band),
      rows = metrics, columns = removal / addition / rewiring
  heatmaps/heatmap_{metric}.pdf
      empirical networks x (noise type, level), colour = mean change
  null_models/null_models_{metric}.pdf
      core networks (rows) x noise types (columns), one line per model
  per_network/{network}_{model}.pdf
      raw metric values vs noise level, baseline as a grey line/band
and the table outputs/global_properties/baseline_vs_perturbed_summary.csv
(mean/std/median of the change per network, model, noise type and level).

Run from the repository root:
  uv run python scripts/global_properties/analysis/plot_baseline_vs_perturbed_all_metrics.py
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

RESULTS_DIR = "./outputs/global_properties/all_metrics/"
FIG_DIR = "./outputs/global_properties/figures/baseline_vs_perturbed/"
SUMMARY_CSV = "./outputs/global_properties/baseline_vs_perturbed_summary.csv"

# metric -> (label, how the change vs baseline is expressed)
METRICS = {
    "gcc": ("GCC fraction", "diff"),
    "num_singletons": ("Singletons", "diff"),
    "efficiency": ("Global efficiency", "rel"),
    "transitivity": ("Transitivity", "rel"),
    "assortativity": ("Degree assortativity", "diff"),
    "max_core": ("Max k-core", "rel"),
    "max_core_frac": ("Max k-core node fraction", "diff"),
    "largest_eigenvalue": ("Largest eigenvalue", "rel"),
    "degree_rank_corr": ("Degree rank corr.", "sim"),
    "degree_top1_overlap": ("Degree top-1% overlap", "sim"),
    "degree_top10_overlap": ("Degree top-10% overlap", "sim"),
    "pagerank_rank_corr": ("PageRank rank corr.", "sim"),
    "pagerank_top1_overlap": ("PageRank top-1% overlap", "sim"),
    "pagerank_top10_overlap": ("PageRank top-10% overlap", "sim"),
}
CHANGE_LABEL = {"rel": "relative change", "diff": "difference", "sim": "similarity to baseline",
                "log2": "log2(perturbed / baseline)"}
CROSS_NETWORK_METRICS = [m for m in METRICS if m != "num_singletons"]

ACTIONS = ["removal", "addition", "rewiring"]
NOISE_TYPES = {
    "removal": ["random_removal", "hub_removal", "periphery_removal", "betweenness_removal"],
    "addition": ["random_addition", "hub_addition", "periphery_addition", "triadic_closure_addition"],
    "rewiring": ["degree_preserving_rewiring"],
}
ALL_NOISE_TYPES = [nt for action in ACTIONS for nt in NOISE_TYPES[action]]

# colour follows the target; the fourth target differs per action and never
# shares a panel with another fourth target, so it gets one slot
TARGET_STYLE = {
    "random": {"color": "#2a78d6", "marker": "o", "label": "Random"},
    "hub": {"color": "#eb6834", "marker": "s", "label": "Hub"},
    "periphery": {"color": "#1baf7a", "marker": "^", "label": "Periphery"},
    "betweenness": {"color": "#6250d6", "marker": "D", "label": "Betweenness"},
    "triadic_closure": {"color": "#6250d6", "marker": "D", "label": "Triadic closure"},
    "degree_preserving": {"color": "#6250d6", "marker": "D", "label": "Degree-preserving"},
}
MODEL_STYLE = {
    "empirical": {"color": "#3a3a38", "label": "Empirical"},
    "er": {"color": "#2a78d6", "label": "Erdős–Rényi"},
    "conf": {"color": "#eb6834", "label": "Configuration"},
    "sbm": {"color": "#1baf7a", "label": "SBM"},
}
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a11d1d"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "gray_red", ["#f0efec", "#ec835a", "#a11d1d", "#4a0b0b"]
)
GRID = "#e1e0d9"
INK_MUTED = "#6b6a66"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------


def load():
    files = sorted(glob.glob(RESULTS_DIR + "*/*/*.parquet"))
    perturbed = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    files = sorted(glob.glob(RESULTS_DIR + "*/*/baseline.csv"))
    baselines = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return perturbed, baselines


def add_changes(perturbed, baselines):
    keys = ["network", "model", "replicate"]
    merged = perturbed.merge(baselines, on=keys, suffixes=("", "_base"), how="left")
    missing = merged[f"{next(iter(METRICS))}_base"].isna().sum()
    if missing:
        print(f"warning: {missing} perturbed networks without baseline")
    for metric, (_, kind) in METRICS.items():
        p, b = merged[metric], merged[f"{metric}_base"]
        if kind == "rel":
            merged[f"d_{metric}"] = (p - b) / b.replace(0, np.nan)
            merged[f"r_{metric}"] = p / b.replace(0, np.nan)
        elif kind == "diff":
            merged[f"d_{metric}"] = p - b
        else:
            merged[f"d_{metric}"] = p
    return merged


def summarise(changes):
    cols = [f"d_{m}" for m in METRICS]
    keys = ["network", "model", "noise_type", "target", "action", "level"]
    grouped = changes.groupby(keys)[cols]
    summary = pd.concat(
        {"mean": grouped.mean(), "std": grouped.std(), "median": grouped.median()}, axis=1
    )
    summary.columns = [f"{m}_{stat}" for stat, m in summary.columns]
    # log2 of the mean ratio (a ratio of 0, i.e. metric vanished, gives NaN)
    rel = [m for m, (_, kind) in METRICS.items() if kind == "rel"]
    with np.errstate(divide="ignore"):
        log2 = np.log2(changes.groupby(keys)[[f"r_{m}" for m in rel]].mean())
    log2 = log2.replace(-np.inf, np.nan)
    log2.columns = [f"log2_{m}_mean" for m in rel]
    # spread of the log2 ratio over repeats/replicates
    with np.errstate(divide="ignore"):
        per_row = np.log2(changes[[f"r_{m}" for m in rel]]).replace(-np.inf, np.nan)
    per_row.columns = [f"log2_{m}_std" for m in rel]
    log2_std = pd.concat([changes[keys], per_row], axis=1).groupby(keys).std()
    return summary.join(log2).join(log2_std).reset_index()


def change_column(metric):
    """Column (without _mean/_std) and kind used in the cross-network figures."""
    kind = METRICS[metric][1]
    return (f"log2_{metric}", "log2") if kind == "rel" else (f"d_{metric}", kind)


def x_values(levels):
    return np.asarray(levels) * 100


def fmt_axis(ax, kind):
    if kind == "rel":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if kind in ("rel", "diff", "log2"):
        ax.axhline(0, color="#8a8984", lw=0.8, zorder=1)
    else:
        ax.axhline(1, color="#8a8984", lw=0.8, zorder=1)


def target_legend(fig, **kwargs):
    """One entry per colour; the fourth slot is named per action."""
    handles = [
        Line2D([0], [0], color=s["color"], marker=s["marker"], ms=5, lw=1.5, label=s["label"])
        for s in (TARGET_STYLE[t] for t in ("random", "hub", "periphery"))
    ]
    s = TARGET_STYLE["betweenness"]
    handles.append(Line2D([0], [0], color=s["color"], marker=s["marker"], ms=5, lw=1.5,
                          label="Betweenness (removal) / triadic closure (addition) / degree-preserving (rewiring)"))
    fig.legend(handles=handles, ncol=len(handles), loc="lower center", **kwargs)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------


def plot_summary(changes, path):
    """Median (IQR band) over empirical networks of the per-network mean change."""
    emp = changes[changes.model == "empirical"]
    per_net = emp.groupby(["network", "noise_type", "target", "action", "level"])[
        [f"d_{m}" for m in CROSS_NETWORK_METRICS]
    ].mean().reset_index()
    n_net = per_net.network.nunique()
    metrics = CROSS_NETWORK_METRICS

    fig, axes = plt.subplots(len(metrics), 3, figsize=(10, 2.0 * len(metrics)), sharey="row")
    for r, metric in enumerate(metrics):
        label, kind = METRICS[metric]
        for c, action in enumerate(ACTIONS):
            ax = axes[r, c]
            for nt in NOISE_TYPES[action]:
                d = per_net[per_net.noise_type == nt]
                q = d.groupby("level")[f"d_{metric}"].quantile([0.25, 0.5, 0.75]).unstack()
                s = TARGET_STYLE[nt.rsplit("_", 1)[0]]
                x = x_values(q.index)
                ax.fill_between(x, q[0.25], q[0.75], color=s["color"], alpha=0.15, lw=0)
                ax.plot(x, q[0.5], color=s["color"], marker=s["marker"], ms=3.5, lw=1.5)
            fmt_axis(ax, kind)
            if r == 0:
                ax.set_title(action.capitalize(), fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{label}\n({CHANGE_LABEL[kind]})")
            if r == len(metrics) - 1:
                ax.set_xlabel("Noise level (% of edges)")
    fig.suptitle(
        f"Change vs baseline across {n_net} empirical networks (median, IQR band)",
        fontweight="bold", y=0.997,
    )
    target_legend(fig, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.015, 1, 0.99))
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)


def plot_heatmap(summary, metric, path):
    label = METRICS[metric][0]
    column, kind = change_column(metric)
    emp = summary[summary.model == "empirical"]
    table = emp.pivot_table(index="network", columns=["noise_type", "level"], values=f"{column}_mean",
                            dropna=False)
    cols = [(nt, lv) for nt in ALL_NOISE_TYPES for lv in sorted(emp.level[emp.noise_type == nt].unique())]
    table = table.reindex(columns=cols).sort_index()
    values = table.to_numpy()

    if kind == "sim":
        cmap, norm = SEQUENTIAL, plt.Normalize(0, 1)
        values = 1 - values
        cbar_label = f"1 − {label} (dissimilarity to baseline)"
    else:
        lim = np.nanpercentile(np.abs(values), 97) if np.isfinite(values).any() else 1
        lim = lim if lim > 0 else 1
        cmap, norm = DIVERGING, TwoSlopeNorm(0, -lim, lim)
        cbar_label = f"{label}: {CHANGE_LABEL[kind]} (clipped at ±{lim:.3g})"

    fig, ax = plt.subplots(figsize=(15, 0.2 * len(table) + 2.2))
    cmap = cmap.with_extremes(bad="#c3c2b7")
    im = ax.imshow(np.clip(values, norm.vmin, norm.vmax), aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.grid(False)
    ax.set_yticks(range(len(table)), table.index, fontsize=7)
    ax.set_xticks(range(len(cols)), [f"{lv * 100:g}" for _, lv in cols], fontsize=6, rotation=90)
    ax.set_xlabel("Noise level (% of edges)")
    # separate and name the noise types
    start = 0
    for nt in ALL_NOISE_TYPES:
        n = sum(1 for c in cols if c[0] == nt)
        if start:
            ax.axvline(start - 0.5, color="white", lw=2)
        ax.text(start + n / 2 - 0.5, -1, nt.replace("_", "\n"), ha="center", va="bottom", fontsize=7)
        start += n
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(cbar_label)
    ax.set_title(f"{label}: perturbed vs baseline (mean over repeats; grey = no data / metric is 0)",
                 fontweight="bold", pad=34)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_null_models(summary, metric, path):
    label = METRICS[metric][0]
    column, kind = change_column(metric)
    cores = sorted(summary.loc[summary.model != "empirical", "network"].unique())
    if not cores:
        return
    fig, axes = plt.subplots(
        len(cores), len(ALL_NOISE_TYPES), figsize=(2.0 * len(ALL_NOISE_TYPES), 2.0 * len(cores)),
        squeeze=False,
    )
    for r, net in enumerate(cores):
        for c, nt in enumerate(ALL_NOISE_TYPES):
            ax = axes[r, c]
            for model, s in MODEL_STYLE.items():
                d = summary[(summary.network == net) & (summary.model == model)
                            & (summary.noise_type == nt)].sort_values("level")
                if d.empty:
                    continue
                x = x_values(d.level)
                m, sd = d[f"{column}_mean"], d[f"{column}_std"].fillna(0)
                ax.fill_between(x, m - sd, m + sd, color=s["color"], alpha=0.15, lw=0)
                ax.plot(x, m, color=s["color"], lw=1.5, marker="o", ms=2.5)
            fmt_axis(ax, kind)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(nt.replace("_", " "), fontsize=8)
            if c == 0:
                ax.set_ylabel(net, fontweight="bold")
            if r == len(cores) - 1:
                ax.set_xlabel("Level (%)", fontsize=8)
    handles = [Line2D([0], [0], color=s["color"], lw=1.5, label=s["label"]) for s in MODEL_STYLE.values()]
    fig.legend(handles=handles, ncol=4, loc="lower center")
    fig.suptitle(f"{label}: {CHANGE_LABEL[kind]}, each model vs its own baseline (mean ± std; "
                 "y-axes not shared)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(path)
    plt.close(fig)


def plot_network(changes, baselines, network, model, path):
    d = changes[(changes.network == network) & (changes.model == model)]
    base = baselines[(baselines.network == network) & (baselines.model == model)]
    agg = d.groupby(["noise_type", "level"])[list(METRICS)].agg(["mean", "std"])

    fig, axes = plt.subplots(len(METRICS), 3, figsize=(10, 1.9 * len(METRICS)), sharey="row")
    for r, (metric, (label, _)) in enumerate(METRICS.items()):
        b_mean, b_std = base[metric].mean(), base[metric].std()
        for c, action in enumerate(ACTIONS):
            ax = axes[r, c]
            ax.axhline(b_mean, color="#8a8984", lw=1, ls="--", zorder=1)
            if np.isfinite(b_std) and b_std > 0:
                ax.axhspan(b_mean - b_std, b_mean + b_std, color="#8a8984", alpha=0.12, lw=0)
            for nt in NOISE_TYPES[action]:
                if nt not in agg.index.get_level_values(0):
                    continue
                a = agg.loc[nt]
                s = TARGET_STYLE[nt.rsplit("_", 1)[0]]
                x = x_values(a.index)
                m, sd = a[(metric, "mean")], a[(metric, "std")].fillna(0)
                ax.fill_between(x, m - sd, m + sd, color=s["color"], alpha=0.15, lw=0)
                ax.plot(x, m, color=s["color"], marker=s["marker"], ms=3.5, lw=1.5)
            if r == 0:
                ax.set_title(action.capitalize(), fontweight="bold")
            if c == 0:
                ax.set_ylabel(label)
            if r == len(METRICS) - 1:
                ax.set_xlabel("Noise level (% of edges)")
    title = network if model == "empirical" else f"{network} — {MODEL_STYLE[model]['label']} null model"
    fig.suptitle(f"{title}: perturbed (mean ± std) vs baseline (dashed)", fontweight="bold", y=0.997)
    target_legend(fig, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.015, 1, 0.99))
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    for sub in ("summary", "heatmaps", "null_models", "per_network"):
        os.makedirs(os.path.join(FIG_DIR, sub), exist_ok=True)

    perturbed, baselines = load()
    changes = add_changes(perturbed, baselines)
    summary = summarise(changes)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"{len(perturbed)} perturbed networks, summary -> {SUMMARY_CSV}")

    plot_summary(changes, os.path.join(FIG_DIR, "summary", "summary_empirical_networks.pdf"))
    for metric in METRICS:
        if metric not in CROSS_NETWORK_METRICS:
            continue
        plot_heatmap(summary, metric, os.path.join(FIG_DIR, "heatmaps", f"heatmap_{metric}.pdf"))
        plot_null_models(summary, metric, os.path.join(FIG_DIR, "null_models", f"null_models_{metric}.pdf"))
    print("summary, heatmaps and null-model figures done")

    for (network, model), _ in changes.groupby(["network", "model"]):
        plot_network(changes, baselines, network, model,
                     os.path.join(FIG_DIR, "per_network", f"{network}_{model}.pdf"))
    print(f"figures in {FIG_DIR}")
