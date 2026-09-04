import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

# --------------------------------------------------------------------------- #
# CONFIG — Matching CSV conventions & row/column layout                      #
# --------------------------------------------------------------------------- #

ROWS = ["PPI", "Astro", "Power", "Wiki"]

COLS_REM = [
    (
        "Random removal",
        "perturbed_random_target",
        ["removed_edges", "removal"],
        "removal",
    ),
    ("Hub removal", "perturbed_hub_target", ["removal"], "removal"),
    ("Periphery removal", "perturbed_periphery_target", ["removal"], "removal"),
]

COLS_ADD = [
    (
        "Random addition",
        "perturbed_random_target",
        ["added_edges", "addition"],
        "addition",
    ),
    ("Hub addition", "perturbed_hub_target", ["addition"], "addition"),
    ("Periphery addition", "perturbed_periphery_target", ["addition"], "addition"),
]

COLS_ALL = COLS_REM + COLS_ADD

CONN_NET = {"PPI": "ppi", "Astro": "astro", "Power": "power", "Wiki": "wiki"}
GCC_NET = {
    "PPI": "chloe_ppi_lcc_2026_02_23",
    "Astro": "ca-AstroPH_gcc",
    "Power": "western_us_power_grid",
    "Wiki": "wiki-Vote_gcc",
}

CONN_ENDPOINT = {"removal": 50.0, "addition": 150.0}
GCC_ENDPOINT = {"removal": 0.5, "addition": 0.5}


def _nearest(series, target):
    """Row index whose noise_level is closest to target."""
    return (series - target).abs().idxmin()


def build_matrices(conn, gcc):
    n_r, n_c = len(ROWS), len(COLS_ALL)
    S = np.full((n_r, n_c), np.nan)  # log2 ratio
    F = np.full((n_r, n_c), np.nan)  # fold change
    G = np.full((n_r, n_c), np.nan)  # endpoint GCC fraction
    base = np.full(n_r, np.nan)  # baseline lambda2 per row

    for i, net in enumerate(ROWS):
        cid, gid = CONN_NET[net], GCC_NET[net]

        # Baseline lambda2 for this network
        b = conn[(conn.network_id == cid) & (conn.noise_type == "baseline")]
        if len(b):
            base[i] = float(b.avg_algebraic_connectivity.iloc[0])

        for j, (_, ntype, conn_ops, gcc_op) in enumerate(COLS_ALL):
            # Endpoint algebraic connectivity
            sub = conn[
                (conn.network_id == cid)
                & (conn.noise_type == ntype)
                & (conn.edge_operation.isin(conn_ops))
            ]
            if len(sub) and not np.isnan(base[i]):
                ep = CONN_ENDPOINT[gcc_op]
                r = sub.loc[_nearest(sub.noise_level, ep)]
                lam = float(r.avg_algebraic_connectivity)
                if lam > 0 and base[i] > 0:
                    F[i, j] = lam / base[i]
                    S[i, j] = np.log2(F[i, j])

            # Endpoint GCC fraction
            gsub = gcc[
                (gcc.network == gid)
                & (gcc.noise_type == ntype)
                & (gcc.action == gcc_op)
            ]
            if len(gsub):
                ep_g = GCC_ENDPOINT[gcc_op]
                gr = gsub.loc[_nearest(gsub.noise_level, ep_g)]
                G[i, j] = float(gr.gcc_mean)

    return S, F, G, base


def fold_label(f):
    if np.isnan(f):
        return "n/a"
    if f >= 100:
        return f"\u00d7{f:.0f}"
    if f >= 10:
        return f"\u00d7{f:.1f}"
    if f >= 1:
        return f"\u00d7{f:.1f}"
    return f"\u00d7{f:.2f}"


def draw(S, F, G, base, out):
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": "#333333",
            "figure.dpi": 300,
        }
    )
    n_r = len(ROWS)

    # Dynamic uncapped scale to reflect actual data range
    smax = np.nanmax(np.abs(S)) if not np.all(np.isnan(S)) else 1.0
    norm = TwoSlopeNorm(vmin=-smax, vcenter=0.0, vmax=smax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e4e4e4")

    fig = plt.figure(figsize=(13.2, 5.8))
    gs = GridSpec(
        1,
        5,
        width_ratios=[1.6, 3.2, 3.2, -0.3, 0.28],
        wspace=0.18,
        left=0.09,
        right=0.92,
        top=0.75,
        bottom=0.10,
    )

    ax_b = fig.add_subplot(gs[0, 0])
    ax_rem = fig.add_subplot(gs[0, 1])
    ax_add = fig.add_subplot(gs[0, 2])
    _ = fig.add_subplot(gs[0, 3])
    _.axis("off")
    ax_c = fig.add_subplot(gs[0, 4])

    S_rem, F_rem, G_rem = S[:, :3], F[:, :3], G[:, :3]
    S_add, F_add, G_add = S[:, 3:], F[:, 3:], G[:, 3:]

    # Heatmaps
    im_rem = ax_rem.imshow(S_rem, cmap=cmap, norm=norm, aspect="auto")
    im_add = ax_add.imshow(S_add, cmap=cmap, norm=norm, aspect="auto")

    # Column Titles
    ax_rem.set_xticks(range(3))
    ax_rem.set_xticklabels(
        [c[0].replace(" ", "\n") for c in COLS_REM],
        fontsize=14.5,  # fontweight="bold"
    )
    ax_rem.xaxis.set_ticks_position("top")
    ax_rem.set_yticks([])
    ax_rem.tick_params(length=0)

    ax_add.set_xticks(range(3))
    ax_add.set_xticklabels(
        [c[0].replace(" ", "\n") for c in COLS_ADD],
        fontsize=14.5,  # fontweight="bold"
    )
    ax_add.xaxis.set_ticks_position("top")
    ax_add.set_yticks([])
    ax_add.tick_params(length=0)

    # White cell gridlines
    for ax_h in (ax_rem, ax_add):
        for k in range(4):
            ax_h.axvline(k - 0.5, color="white", lw=1.5)
        for k in range(n_r + 1):
            ax_h.axhline(k - 0.5, color="white", lw=1.5)

    # Group Header Titles
    ax_rem.annotate(
        "EDGE REMOVAL (50%)",
        xy=(1, -1.05),
        xycoords=("data", "data"),
        ha="center",
        va="bottom",
        fontsize=15.5,
        fontweight="bold",
        color="#222222",
        annotation_clip=False,
    )
    ax_add.annotate(
        "EDGE ADDITION (50%)",
        xy=(1, -1.05),
        xycoords=("data", "data"),
        ha="center",
        va="bottom",
        fontsize=15.5,
        fontweight="bold",
        color="#222222",
        annotation_clip=False,
    )

    # Cell Text Annotations + GCC Dots
    gmin, gmax = 0.0, 1.0
    dot_min, dot_max = 16, 360

    def annotate_cells(ax_h, S_sub, F_sub, G_sub):
        for i in range(n_r):
            for j in range(3):
                val_s = S_sub[i, j]
                val_f = F_sub[i, j]
                val_g = G_sub[i, j]

                if not np.isnan(val_s):
                    rgba = cmap(norm(val_s))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    tcol = "white" if lum < 0.52 else "#111111"
                    ax_h.text(
                        j,
                        i - 0.12,
                        fold_label(val_f),
                        ha="center",
                        va="center",
                        fontsize=14.5,
                        fontweight="bold",
                        color=tcol,
                    )
                if not np.isnan(val_g):
                    frac = np.clip((val_g - gmin) / (gmax - gmin), 0, 1)
                    area = dot_min + frac * (dot_max - dot_min)
                    ax_h.scatter(
                        j,
                        i + 0.26,
                        s=area,
                        color="#222222",
                        alpha=0.80,
                        edgecolors="white",
                        linewidths=0.9,
                        zorder=5,
                    )

    annotate_cells(ax_rem, S_rem, F_rem, G_rem)
    annotate_cells(ax_add, S_add, F_add, G_add)

    # Baseline Horizontal Bar Strip
    ypos = np.arange(n_r)
    floor = np.nanmin(base) / 4.0 if not np.all(np.isnan(base)) else 1e-4
    ax_b.barh(
        ypos,
        base,
        left=floor,
        height=0.5,
        color="#B6D9F1",
        edgecolor="#2c3a52",
        zorder=3,
    )
    ax_b.set_xscale("log")
    ax_b.set_ylim(n_r - 0.5, -0.5)
    ax_b.set_yticks(range(n_r))
    ax_b.set_yticklabels(ROWS, fontsize=14.5, fontweight="bold")
    ax_b.tick_params(axis="y", length=0, pad=6)
    ax_b.set_xlabel(r"baseline $\lambda_2^{\,0}$", fontsize=14.5, fontweight="bold")
    ax_b.xaxis.set_label_position("top")
    ax_b.xaxis.tick_top()
    ax_b.tick_params(axis="x", labelsize=13.5)
    ax_b.set_xlim(floor, np.nanmax(base) * 8 if not np.all(np.isnan(base)) else 1.0)
    ax_b.grid(axis="x", which="major", color="#e0e0e0", lw=0.6, zorder=0)

    for i, v in enumerate(base):
        if not np.isnan(v):
            v_str = f"{v:.2e}".replace("e-0", "e-") if v < 0.005 else f"{v:.3f}"
            ax_b.text(
                v * 1.25,
                i,
                v_str,
                va="center",
                ha="left",
                fontsize=14.0,
                # fontweight="bold",
                color="#222222",
                zorder=4,
            )
    for s in ("top", "right", "left"):
        ax_b.spines[s].set_visible(False)

    # Colorbar
    cb = fig.colorbar(im_add, cax=ax_c)
    cb.set_label(
        r"Signed spectral response"
        "\n"
        r"$s_{\lambda_2} = \log_2(\lambda_2 / \lambda_2^{\,0})$",
        fontsize=14.5,
        fontweight="bold",
        labelpad=10,
    )
    cb.ax.tick_params(labelsize=14.5)
    gs = GridSpec(
        1,
        5,
        width_ratios=[1.6, 3.2, 3.2, 10, 0.25],  # 0.06 controls the gap width
        wspace=0.18,
        left=0.09,
        right=0.92,
        top=0.75,
        bottom=0.10,
    )

    # Framed GCC Legend Box (upper right)
    leg_ax = fig.add_axes([0.26, 0.92, 0.50, 0.075])
    leg_ax.set_facecolor("#fafafa")
    for spine in leg_ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#b0b0b0")
        spine.set_linewidth(1.0)
    leg_ax.set_xticks([])
    leg_ax.set_yticks([])

    ref_vals = [0.5, 0.8, 1.0]
    leg_ax.text(
        0.04,
        0.5,
        "Endpoint GCC fraction:",
        ha="left",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color="#222222",
    )
    x_starts = [0.55, 0.70, 0.85]
    for k, rv in enumerate(ref_vals):
        frac = (rv - gmin) / (gmax - gmin)
        area = dot_min + frac * (dot_max - dot_min)
        leg_ax.scatter(
            x_starts[k],
            0.5,
            s=area,
            color="#222222",
            alpha=0.80,
            edgecolors="white",
            linewidths=0.9,
        )
        leg_ax.text(
            x_starts[k] + 0.04,
            0.5,
            f"{rv:.1f}",
            ha="left",
            va="center",
            fontsize=14.5,
            fontweight="bold",
            color="#333333",
        )
    leg_ax.set_xlim(0.0, 1.0)
    leg_ax.set_ylim(0.0, 1.0)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote figure to {out}")


def main():
    df_conn = pd.read_csv(
        "outputs/global_properties/algebraic_connectivity/aggregated_mean_fiedler_values_perturbed_plus_baseline.csv"
    )
    df_gcc = pd.read_csv("outputs/global_properties/aggregated_gcc_singletons.csv")

    S, F, G, base = build_matrices(df_conn, df_gcc)
    draw(
        S,
        F,
        G,
        base,
        "outputs/global_properties/algebraic_connectivity/figures/overview/algebraic_connectivity_overview.pdf",
    )


if __name__ == "__main__":
    main()
