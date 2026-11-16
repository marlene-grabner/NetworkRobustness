"""
One combined overview figure per empirical network: 3 rows (hub/periphery/
random target) x 4 columns (community-detection algorithm), each subplot
overlaying the three null-model comparison curves (er/conf/sbm) for
diff_vs_baseline_ari_vs_null = empirical_vs_baseline_ari - null_vs_baseline_ari.

Reads outputs/local_structure/overview_csvs/perturbed/ari_diff_vs_null_models.csv,
produced by normalize_ari_vs_null_models.py.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

INPUT_PATH = "outputs/local_structure/overview_csvs/perturbed/ari_diff_vs_null_models.csv"
OUTPUT_DIR = "outputs/local_structure/figures/perturbed/null_diff"

networks = ["astro", "ppi", "power", "wiki"]
targets = ["hub", "periphery", "random"]
algorithms = ["infomap", "leiden", "louvain", "label_propagation"]

target_labels = {"hub": "Hub Target", "periphery": "Periphery Target", "random": "Random Target"}
algorithm_labels = {
    "infomap": "Infomap", "leiden": "Leiden",
    "louvain": "Louvain", "label_propagation": "Label Propagation",
}
network_labels = {"astro": "Astrophysics Collaboration", "ppi": "Protein Interaction Network",
                   "power": "Western US Power Grid", "wiki": "Wikipedia Vote"}

null_styles = {
    "er": {"color": "#1b9e77", "label": "Erdos-Rényi"},
    "conf": {"color": "#d95f02", "label": "Configuration"},
    "sbm": {"color": "#7570b3", "label": "SBM"},
}

full_x_ticks = [50, 60, 70, 80, 100, 125, 150, 200, 250, 300]
tick_formatter = FuncFormatter(lambda val, pos: f"{int(val)}")


def relative_size(row):
    if row["target"] == "baseline" or row["level"] == 0:
        return 100.0
    level_pct = row["level"] * 100
    return 100.0 - level_pct if row["type_of_noise"] == "removal" else 100.0 + level_pct


def plot_network(df, network, save_fig):
    df_net = df[df["network_id"] == network].copy()
    if df_net.empty:
        print(f"No ARI null-diff data for {network}, skipping.")
        return

    df_net["relative_size"] = df_net.apply(relative_size, axis=1)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.1, top=0.90, bottom=0.12)
    fig.suptitle(f"{network_labels[network]} — ARI vs. own baseline, empirical minus null model",
                 fontsize=20, fontweight="bold")

    for row_idx, target in enumerate(targets):
        baseline_row = df_net[df_net["target"] == "baseline"].copy()

        for col_idx, alg in enumerate(algorithms):
            ax = axes[row_idx, col_idx]
            df_cell = df_net[(df_net["target"] == target) & (df_net["algorithm"] == alg)]
            base_cell = baseline_row[baseline_row["algorithm"] == alg]

            for null_model, style in null_styles.items():
                df_line = df_cell[df_cell["null_model"] == null_model].copy()
                base_line = base_cell[base_cell["null_model"] == null_model].copy()

                if not base_line.empty:
                    base_line["relative_size"] = 100.0
                    df_line = pd.concat([df_line, base_line])

                df_line = df_line.sort_values("relative_size").dropna(subset=["diff_vs_baseline_ari_vs_null"])
                if df_line.empty:
                    continue

                x = df_line["relative_size"]
                y = df_line["diff_vs_baseline_ari_vs_null"]
                std = df_line["std_diff_vs_baseline_ari_vs_null"]

                ax.plot(x, y, color=style["color"], linewidth=2, marker="o", markersize=4)
                ax.fill_between(x, y - std, y + std, color=style["color"], alpha=0.15)

            ax.axhline(0.0, color="gray", linestyle="-", alpha=0.4, linewidth=1.2)
            ax.axvline(100, color="gray", linestyle="--", alpha=0.4, linewidth=1.2)
            ax.set_xscale("log")
            ax.set_xticks(full_x_ticks)
            ax.set_xticks([], minor=True)
            ax.xaxis.set_major_formatter(tick_formatter)

            if row_idx == 0:
                ax.set_title(algorithm_labels[alg], fontsize=15, fontweight="bold", pad=12)
            if col_idx == 0:
                ax.set_ylabel(f"{target_labels[target]}\nΔ ARI vs base", fontsize=12, fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel("Relative Network Size (%)", fontsize=11, labelpad=8)

            ax.grid(True, which="major", axis="both", color="#E0E0E0", linestyle="--", alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color("#CCCCCC")

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
    df = pd.read_csv(INPUT_PATH)
    for network in networks:
        plot_network(df, network, f"{OUTPUT_DIR}/{network}_ari_null_diff_overview.pdf")


if __name__ == "__main__":
    main()
