"""
Panel (b): baseline topological descriptors of the null ensembles,
expressed relative to their empirical parent network.

Main heatmap : log2(null / empirical) for strictly positive descriptors.
Bottom strip : degree assortativity, shown as a signed difference
               (r_null - r_empirical) because the log ratio is undefined
               for a descriptor that changes sign.
"""

import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

EMPIRICAL_CSV = "outputs/models/baseline_network_properties.csv"
NULL_CSV = "outputs/models/null_model_network_properties.csv"
OUTFILE = "outputs/models/figures/network_topology/null_ratio_heatmap.pdf"

# (short key, column in EMPIRICAL_CSV, prefix in NULL_CSV, display name)
PARENTS = [
    ("ppi", "Protein-Protein Interaction Network", "PPI", "PPI"),
    ("power", "Western US Power Grid Network", "Power Grid", "Power Grid"),
    ("astro", "Astrophysics Collaboration Network", "Astrophysics", "AstroPh"),
    ("wiki", "Wikipedia Vote Network", "Wikipedia", "Wiki-Vote"),
]

MODELS = ["ER", "Conf", "SBM"]
MODEL_LABELS = {"ER": "ER", "Conf": "Conf.", "SBM": "SBM"}

# row  : exact index label in both CSVs
# mode : "ratio" -> log2(null/emp); "diff" -> null - emp, drawn separately
DESCRIPTORS = [
    dict(key="nodes_full", label=r"$N$", row="Nodes_Full", mode="ratio"),
    dict(key="edges_full", label=r"$M$", row="Edges_Full", mode="ratio"),
    dict(key="gccfrac", label="GCC frac.", row="GCC_Fraction", mode="ratio"),
    dict(
        key="meandeg", label=r"$\langle k \rangle$", row="Average_Degree", mode="ratio"
    ),
    dict(key="density", label=r"$\rho$", row="Density", mode="ratio"),
    dict(key="cv", label=r"$CV_k$", row="Degree_CV", mode="ratio"),
    dict(key="skew", label="Skew.", row="Degree_Skew", mode="ratio"),
    dict(key="clust", label=r"$\bar{C}$", row="Clustering_Coefficient", mode="ratio"),
    dict(key="trans", label="Transit.", row="Transitivity", mode="ratio"),
    dict(key="mod", label=r"$Q$", row="Modularity_Louvain", mode="ratio"),
    dict(key="lam2", label=r"$\lambda_2$", row="Algebraic_Connectivity", mode="ratio"),
    dict(key="gap", label=r"$\Delta\lambda$", row="Spectral_Gap", mode="ratio"),
    dict(key="path", label=r"$\langle d \rangle$", row="Avg_Path_Length", mode="ratio"),
    dict(key="assort", label=r"$r$", row="Assortativity", mode="diff"),
]

VMAX_LOG2 = 4.0  # colour cap: 4 == a factor of 16 either way
CELL_FONT = 5.4

# ----------------------------------------------------------------------


def load(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def extract(df, column, source_name):
    col, out = df[column], {}
    for d in DESCRIPTORS:
        if d["row"] in df.index:
            out[d["key"]] = float(col[d["row"]])
        else:
            print(f"  ! row '{d['row']}' missing for {source_name}")
            out[d["key"]] = np.nan
    return out


def fmt_ratio(x):
    if 0.01 <= abs(x) < 100:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.0e}".replace("e-0", "e-").replace("e+0", "e")


def draw(ax, M, raw, mode, vmax, cmap):
    im = ax.imshow(
        np.clip(M, -vmax, vmax),
        cmap=cmap,
        aspect="auto",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
    )
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, facecolor="0.9", edgecolor="none"
                    )
                )
                ax.text(
                    j,
                    i,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize=CELL_FONT,
                    color="0.45",
                )
                continue
            strong = abs(np.clip(M[i, j], -vmax, vmax)) > 0.62 * vmax
            txt = f"×{fmt_ratio(raw[i, j])}" if mode == "ratio" else f"{raw[i, j]:+.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=CELL_FONT,
                color="white" if strong else "black",
            )
    return im


def main():
    emp_df, null_df = load(EMPIRICAL_CSV), load(NULL_CSV)

    for _, col, _, _ in PARENTS:
        if col not in emp_df.columns:
            sys.exit(f"'{col}' not in {EMPIRICAL_CSV}: {list(emp_df.columns)}")
    for _, _, prefix, _ in PARENTS:
        for m in MODELS:
            if f"{prefix} ({m})" not in null_df.columns:
                sys.exit(f"'{prefix} ({m})' not in {NULL_CSV}: {list(null_df.columns)}")

    emp = {key: extract(emp_df, col, disp) for key, col, _, disp in PARENTS}
    null = {
        (key, m): extract(null_df, f"{prefix} ({m})", f"{disp} ({m})")
        for key, _, prefix, disp in PARENTS
        for m in MODELS
    }

    order = [(k, m) for k, _, _, _ in PARENTS for m in MODELS]  # rows
    ratio_desc = [d for d in DESCRIPTORS if d["mode"] == "ratio"]
    diff_desc = [d for d in DESCRIPTORS if d["mode"] == "diff"]

    def build(descs, mode):
        """Rows = null graphs, columns = descriptors."""
        M = np.full((len(order), len(descs)), np.nan)
        raw = np.full_like(M, np.nan)
        for i, (parent, model) in enumerate(order):
            for j, d in enumerate(descs):
                e, n = emp[parent][d["key"]], null[(parent, model)][d["key"]]
                if not (np.isfinite(e) and np.isfinite(n)):
                    continue
                if mode == "ratio":
                    if e <= 0 or n <= 0:
                        print(
                            f"  ! non-positive {d['key']} for {parent}/{model} "
                            f"(emp={e:g}, null={n:g}) -> masked"
                        )
                        continue
                    raw[i, j], M[i, j] = n / e, np.log2(n / e)
                else:
                    raw[i, j] = M[i, j] = n - e
        return M, raw

    R, R_raw = build(ratio_desc, "ratio")
    D, D_raw = build(diff_desc, "diff")

    # ------------------------------------------------------------------
    mpl.rcParams.update(
        {"font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42}
    )

    n_ratio, n_diff = R.shape[1], D.shape[1]
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[n_ratio, n_diff],
        wspace=0.05,
        left=0.155,
        right=0.905,
        top=0.755,
        bottom=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1], sharey=ax)

    vmax = max(
        min(VMAX_LOG2, np.nanmax(np.abs(R)) if np.isfinite(R).any() else 1.0), 1e-9
    )
    dmax = max(np.nanmax(np.abs(D)) if np.isfinite(D).any() else 1.0, 1e-9)

    im = draw(ax, R, R_raw, "ratio", vmax, "RdBu_r")
    im_d = draw(ax_d, D, D_raw, "diff", dmax, "PuOr_r")

    for a, descs in ((ax, ratio_desc), (ax_d, diff_desc)):
        a.set_xticks(range(len(descs)))
        a.set_xticklabels([d["label"] for d in descs], fontsize=7.5)
        a.xaxis.set_ticks_position("top")
        a.tick_params(axis="both", length=0)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([MODEL_LABELS[m] for _, m in order], fontsize=7)
    plt.setp(ax_d.get_yticklabels(), visible=False)

    # parent-network grouping: separators plus a bracketed label per group
    for g in range(1, len(PARENTS)):
        for a in (ax, ax_d):
            a.axhline(g * len(MODELS) - 0.5, color="white", lw=2.2)
    for g, (_, _, _, disp) in enumerate(PARENTS):
        y = g * len(MODELS) + (len(MODELS) - 1) / 2
        ax.text(
            -0.075,
            y,
            disp,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    p, p_d = ax.get_position(), ax_d.get_position()

    # main scale: horizontal bar above the ratio block
    cb = fig.colorbar(
        im, orientation="horizontal", cax=fig.add_axes([p.x0, 0.905, p.width, 0.028])
    )
    ticks = np.arange(-np.floor(vmax), np.floor(vmax) + 1)
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels(
        [f"×{2.0**t:g}" if t >= 0 else f"×1/{2.0**-t:g}" for t in ticks], fontsize=6.5
    )
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.set_xlabel(
        r"$\log_2(\mathrm{Null}\,/\,\mathrm{Empirical\ parent})$",
        fontsize=7.5,
        labelpad=3,
    )

    # signed scale: vertical bar beside the assortativity column
    cb_d = fig.colorbar(
        im_d, cax=fig.add_axes([p_d.x1 + 0.012, p_d.y0, 0.013, p_d.height])
    )
    cb_d.set_label(r"$\Delta r$", fontsize=7.5, labelpad=2)
    cb_d.ax.tick_params(labelsize=6.5)

    fig.savefig(OUTFILE, bbox_inches="tight")
    fig.savefig(OUTFILE.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {OUTFILE}")


if __name__ == "__main__":
    main()
