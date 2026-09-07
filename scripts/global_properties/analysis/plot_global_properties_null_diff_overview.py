"""
One combined overview figure per empirical network: 3 rows (hub/periphery/
random target) x 3 columns (Fiedler value, GCC fraction, global efficiency),
each subplot overlaying the three null-model comparison curves (er/conf/sbm).

Fiedler uses log2(empirical/null) (strictly positive, multiplicative metric);
GCC fraction and global efficiency use empirical - null (bounded [0, 1]).

Reads:
    outputs/global_properties/algebraic_connectivity/aggregated_fiedler_log2_ratio_vs_null_models.csv
    outputs/global_properties/gcc_singletons/aggregated_gcc_diff_vs_null_models.csv
    outputs/global_properties/global_efficiency/global_efficiency_diff_vs_null_models.csv (optional, may not exist yet)
produced by the normalize_*_vs_null_models.py scripts.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from pathlib import Path

FIEDLER_PATH = "outputs/global_properties/algebraic_connectivity/aggregated_fiedler_log2_ratio_vs_null_models.csv"
GCC_PATH = "outputs/global_properties/gcc_singletons/aggregated_gcc_diff_vs_null_models.csv"
EFFICIENCY_PATH = "outputs/global_properties/global_efficiency/global_efficiency_diff_vs_null_models.csv"
OUTPUT_DIR = "outputs/global_properties/figures/null_model_diff"

networks = ["astro", "ppi", "power", "wiki"]
targets = ["hub", "periphery", "random"]
target_labels = {"hub": "Hub Target", "periphery": "Periphery Target", "random": "Random Target"}
network_labels = {"astro": "Astrophysics Collaboration", "ppi": "Protein Interaction Network",
                   "power": "Western US Power Grid", "wiki": "Wikipedia Vote"}

null_styles = {
    "er": {"color": "#1b9e77", "label": "Erdos-Rényi"},
    "conf": {"color": "#d95f02", "label": "Configuration"},
    "sbm": {"color": "#7570b3", "label": "SBM"},
}

full_x_ticks = [50, 60, 70, 80, 100, 125, 150, 200, 250, 300]
tick_formatter = FuncFormatter(lambda val, pos: f"{int(val)}")


def strip_target(noise_type):
    if noise_type == "baseline":
        return "baseline"
    return noise_type.removeprefix("perturbed_").removesuffix("_target")


def load_fiedler():
    df = pd.read_csv(FIEDLER_PATH)
    df["target"] = df["noise_type"].apply(strip_target)
    # noise_level here is already on the 100 +/- N% scale
    df = df.rename(columns={"noise_level": "relative_size", "log2_ratio_vs_null": "value",
                             "std_log2_ratio_vs_null": "std"})
    return df[["network_id", "null_model", "target", "relative_size", "value", "std"]]


def load_gcc():
    df = pd.read_csv(GCC_PATH)
    df["target"] = df["noise_type"].apply(strip_target)
    df["relative_size"] = df.apply(
        lambda r: 100.0 if r["target"] == "baseline" else
        (100.0 - r["noise_level"] * 100 if r["action"] == "removal" else 100.0 + r["noise_level"] * 100),
        axis=1,
    )
    df = df.rename(columns={"diff_gcc_vs_null": "value", "std_diff_gcc_vs_null": "std"})
    return df[["network_id", "null_model", "target", "relative_size", "value", "std"]]


def load_efficiency():
    if not Path(EFFICIENCY_PATH).exists():
        return None
    df = pd.read_csv(EFFICIENCY_PATH)
    df["target"] = df["noise_type"]  # already short form (hub/periphery/random/baseline)
    df["relative_size"] = df.apply(
        lambda r: 100.0 if r["target"] == "baseline" else
        (100.0 - r["noise_level"] * 100 if r["action"] == "removal" else 100.0 + r["noise_level"] * 100),
        axis=1,
    )
    df = df.rename(columns={"diff_global_efficiency_vs_null": "value", "std_diff_global_efficiency_vs_null": "std"})
    return df[["network_id", "null_model", "target", "relative_size", "value", "std"]]


columns = [
    ("Fiedler value", r"$\log_2$(empirical / null)", load_fiedler()),
    ("GCC fraction", "Δ GCC fraction", load_gcc()),
    ("Global efficiency", "Δ global efficiency", load_efficiency()),
]


def plot_network(network, save_fig):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.3, top=0.90, bottom=0.12, left=0.09)
    fig.suptitle(f"{network_labels[network]} — empirical minus null model",
                 fontsize=20, fontweight="bold")

    for row_idx, target in enumerate(targets):
        for col_idx, (title, ylabel, df) in enumerate(columns):
            ax = axes[row_idx, col_idx]

            if df is None:
                ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                        fontsize=13, color="gray", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#CCCCCC")
                if row_idx == 0:
                    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
                continue

            df_net = df[df["network_id"] == network]
            baseline_row = df_net[df_net["target"] == "baseline"]
            df_cell = df_net[df_net["target"] == target]

            for null_model, style in null_styles.items():
                df_line = df_cell[df_cell["null_model"] == null_model].copy()
                base_line = baseline_row[baseline_row["null_model"] == null_model].copy()

                if not base_line.empty:
                    base_line = base_line.copy()
                    base_line["relative_size"] = 100.0
                    df_line = pd.concat([df_line, base_line])

                df_line = df_line.sort_values("relative_size").dropna(subset=["value"])
                if df_line.empty:
                    continue

                x = df_line["relative_size"]
                y = df_line["value"]
                std = df_line["std"]

                ax.plot(x, y, color=style["color"], linewidth=2, marker="o", markersize=4)
                ax.fill_between(x, y - std, y + std, color=style["color"], alpha=0.15)

            ax.axhline(0.0, color="gray", linestyle="-", alpha=0.4, linewidth=1.2)
            ax.axvline(100, color="gray", linestyle="--", alpha=0.4, linewidth=1.2)
            ax.set_xscale("log")
            ax.set_xticks(full_x_ticks)
            ax.set_xticks([], minor=True)
            ax.xaxis.set_major_formatter(tick_formatter)

            if row_idx == 0:
                ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
            if row_idx == 2:
                ax.set_xlabel("Relative Network Size (%)", fontsize=11, labelpad=8)

            ax.set_ylabel(ylabel, fontsize=11)

            ax.grid(True, which="major", axis="both", color="#E0E0E0", linestyle="--", alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color("#CCCCCC")

        fig.text(0.02, axes[row_idx, 0].get_position().y0 + axes[row_idx, 0].get_position().height / 2,
                  target_labels[target], fontsize=13, fontweight="bold", rotation=90,
                  ha="center", va="center")

    legend_elements = [
        Line2D([0], [0], color=style["color"], lw=2.5, marker="o", markersize=7, label=style["label"])
        for style in null_styles.values()
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, frameon=False,
               fontsize=14, bbox_to_anchor=(0.5, 0.02))

    plt.savefig(save_fig, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figure saved to {save_fig}")


def main():
    for network in networks:
        plot_network(network, f"{OUTPUT_DIR}/{network}_global_properties_null_diff_overview.pdf")


if __name__ == "__main__":
    main()
