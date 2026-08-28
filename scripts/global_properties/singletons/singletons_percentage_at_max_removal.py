from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
INPUT_CSV = Path("outputs/global_properties/aggregated_gcc_singletons.csv")
OUTPUT_DIR = Path(
    "outputs/global_properties/gcc_singletons/figures/num_singletons_after_removal"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


BASELINE_NODES = {
    "ppi": 16539,
    "power": 4941,
    "astro": 17903,
    "wiki": 7066,
}

NETWORK_LABEL_MAP = {
    "ppi": "PPI",
    "power": "Power Grid",
    "astro": "AstroPh",
    "wiki": "Wiki-Vote",
}

# Crisp Base Network Colors
NETWORK_COLORS = {
    "PPI": "#FAA0A0",  # Red
    "Power Grid": "#A7C7E7",  # Blue
    "AstroPh": "#C1E1C1",  # Green
    "Wiki-Vote": "#FAC898",  # Golden Amber
}

# Target Marker Shapes
TARGET_MARKERS = {"Random": "o", "Hub": "s", "Periphery": "^"}

TARGET_LABEL_MAP = {
    "random": "Random",
    "hub": "Hub",
    "periphery": "Periphery",
    "perturbed_random_target": "Random",
    "perturbed_hub_target": "Hub",
    "perturbed_periphery_target": "Periphery",
}


# ==============================================================================
# DATA PROCESSING
# ==============================================================================
def parse_network_key(net_str: str) -> str:
    s = str(net_str).lower().strip()
    if any(k in s for k in ["astro", "ca-astroph"]):
        return "astro"
    elif any(k in s for k in ["ppi", "chloe"]):
        return "ppi"
    elif any(k in s for k in ["power", "western"]):
        return "power"
    elif any(k in s for k in ["wiki"]):
        return "wiki"
    return "unknown"


def load_and_process_singletons(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["action"] = df["action"].astype(str).str.lower().str.strip()
    df["noise_level"] = pd.to_numeric(df["noise_level"], errors="coerce")

    if df["noise_level"].max() > 2.0:
        df["noise_level"] /= 100.0

    mask = (df["action"] == "removal") & np.isclose(df["noise_level"], 0.5)
    filtered = df[mask].copy()

    filtered["base_key"] = filtered["network"].apply(parse_network_key)
    filtered = filtered[filtered["base_key"] != "unknown"].copy()

    filtered = filtered[
        ~filtered["network"].str.contains("_er|_conf|_sbm", case=False, regex=True)
    ]

    filtered["target_clean"] = filtered["noise_type"].astype(str).map(TARGET_LABEL_MAP)
    filtered["net_label"] = filtered["base_key"].map(NETWORK_LABEL_MAP)

    filtered["baseline_N"] = filtered["base_key"].map(BASELINE_NODES)
    filtered["singletons_mean"] = pd.to_numeric(
        filtered["num_singletons_mean"], errors="coerce"
    )
    filtered["pct_singletons"] = (
        filtered["singletons_mean"] / filtered["baseline_N"]
    ) * 100.0

    grouped = filtered.groupby(
        ["net_label", "target_clean", "base_key"], as_index=False
    )["pct_singletons"].mean()

    grouped["full_label"] = grouped["net_label"] + " (" + grouped["target_clean"] + ")"
    grouped = grouped.sort_values(by="pct_singletons", ascending=True).reset_index(
        drop=True
    )

    return grouped


# ==============================================================================
# PLOTTING PIPELINE
# ==============================================================================
def plot_singletons_lollipop_clean():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
        }
    )

    if not INPUT_CSV.exists():
        print(f"❌ Input CSV not found at: {INPUT_CSV.resolve()}")
        return

    plot_df = load_and_process_singletons(INPUT_CSV)

    if plot_df.empty:
        print("⚠️ No data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(4.4, 4.2), dpi=300)
    y_positions = np.arange(len(plot_df))

    # --- Draw Clean Stems & Shape-Encoded Head Markers ---
    for idx, row in plot_df.iterrows():
        color = NETWORK_COLORS.get(row["net_label"], "#334155")
        marker = TARGET_MARKERS.get(row["target_clean"], "o")
        val = row["pct_singletons"]

        # Solid Stem
        ax.hlines(
            y=idx, xmin=0, xmax=val, color=color, alpha=0.75, linewidth=1.8, zorder=2
        )

        # Head Marker (Shape = Target, Color = Network)
        ax.plot(
            val,
            idx,
            marker=marker,
            color=color,
            markersize=7.5,
            markeredgecolor="#1E293B",
            markeredgewidth=0.8,
            zorder=3,
        )

        # Value Label
        ax.text(
            val + 2,
            idx,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
            color="#334155",
        )

    # --- Axes Formatting ---
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["full_label"])
    ax.set_xlabel(
        "Isolated Singletons (% of Baseline $N$)", labelpad=8, fontweight="bold"
    )

    max_val = plot_df["pct_singletons"].max()
    ax.set_xlim(0, max_val * 1.15)
    ax.set_ylim(-0.6, len(plot_df) - 0.4)

    ax.grid(True, axis="x", linestyle=":", alpha=0.6, color="#94A3B8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    # --- LEGEND 1: Network Family (Colors) ---
    handles_net = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            label=net,
            markersize=8,
            markeredgecolor="#1E293B",
            markeredgewidth=0.6,
        )
        for net, color in NETWORK_COLORS.items()
    ]
    leg_net = ax.legend(
        handles=handles_net,
        title="Network",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=8,
        title_fontsize=8.5,
    )

    # --- LEGEND 2: Target Strategy (Shapes) ---
    handles_target = [
        plt.Line2D(
            [0],
            [0],
            marker=shape,
            color="w",
            markerfacecolor="#64748B",
            label=tgt,
            markersize=7.5,
            markeredgecolor="#1E293B",
            markeredgewidth=0.6,
        )
        for tgt, shape in TARGET_MARKERS.items()
    ]
    leg_tgt = ax.legend(
        handles=handles_target,
        title="Target",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=8,
        title_fontsize=8.5,
        ncols=3,
    )

    ax.add_artist(leg_net)

    plt.tight_layout()

    out_png = OUTPUT_DIR / "singletons_50pct_removal_clean_symbols.png"
    out_pdf = OUTPUT_DIR / "singletons_50pct_removal_clean_symbols.pdf"

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(
        f"✅ Clean Symbol Lollipop Chart saved to:\n   - {out_png.resolve()}\n   - {out_pdf.resolve()}"
    )
    plt.show()


if __name__ == "__main__":
    plot_singletons_lollipop_clean()
