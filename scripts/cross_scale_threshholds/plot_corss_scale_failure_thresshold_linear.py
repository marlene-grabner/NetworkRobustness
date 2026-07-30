from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# KEYWORD TOGGLE
# ==============================================================================
# Choose scale mode: "absolute"  OR  "log_fold"
SCALE_MODE = "absolute"  

RETENTION_THRESHOLD = 0.50  # 50% baseline retention threshold
LOCAL_K_TARGET = "50"
INCLUDE_NULL_MODELS = False # Set True to include ER / Config / SBM rows

# Maximum tested perturbation noise levels in your dataset:
MAX_TESTED_REMOVAL = 50.0   # 50% of edges removed
MAX_TESTED_ADDITION = 200.0 # +200% edges added (300% relative network size)

# Paths to your CSV files (update filenames as needed)
PATHS = {
    "gcc": Path("outputs/global_properties/aggregated_gcc_singletons.csv"),
    "fiedler": Path("outputs/global_properties/algebraic_connectivity/aggregated_mean_fiedler_values_perturbed_plus_baseline.csv"),
    "infomap": Path("outputs/local_structure/overview_csvs/perturbed/infomap_aggregated.csv"),
    "louvain": Path("outputs/local_structure/overview_csvs/perturbed/louvain_aggregated.csv"),
    "leiden": Path("outputs/local_structure/overview_csvs/perturbed/leiden_aggregated.csv"),
    "label_propagation": Path("outputs/local_structure/overview_csvs/perturbed/label_propagation_aggregated.csv"),
    "seed_expansion": Path("outputs/seed_expansion/expansion/perturbed_summary_csvs/local_results_aggregated.csv"),
}

NETWORK_MAP = {
    "astro": ("Astro", "Empirical"), "ca-astroph_gcc": ("Astro", "Empirical"),
    "astro_er": ("Astro", "Erdos-Renyi"), "astro_conf": ("Astro", "Config-Model"), "astro_sbm": ("Astro", "SBM"),
    
    "ppi": ("PPI", "Empirical"), "chloe_ppi_lcc_2026_02_23": ("PPI", "Empirical"),
    "ppi_er": ("PPI", "Erdos-Renyi"), "ppi_conf": ("PPI", "Config-Model"), "ppi_sbm": ("PPI", "SBM"),
    
    "power": ("Power Grid", "Empirical"), "western_us_power_grid": ("Power Grid", "Empirical"),
    "power_er": ("Power Grid", "Erdos-Renyi"), "power_conf": ("Power Grid", "Config-Model"), "power_sbm": ("Power Grid", "SBM"),
    
    "wiki": ("Wiki", "Empirical"), "wiki-vote_gcc": ("Wiki", "Empirical"),
    "wiki_er": ("Wiki", "Erdos-Renyi"), "wiki_conf": ("Wiki", "Config-Model"), "wiki_sbm": ("Wiki", "SBM"),
}


# ==============================================================================
# UNIT STANDARDIZATION & FIXED THRESHOLD COMPUTATION
# ==============================================================================
def parse_network(raw_str: str) -> tuple[str, str]:
    s = str(raw_str).lower().strip()
    for raw_key, (net_name, model_type) in NETWORK_MAP.items():
        if raw_key == s or raw_key in s:
            return net_name, model_type
    return "Unknown", "Unknown"


def standardize_noise_pct(raw_val: float, action: str, is_fiedler: bool = False) -> float:
    """Converts varied CSV inputs to absolute noise percentage modified."""
    if pd.isnull(raw_val):
        return np.nan
    raw_val = float(raw_val)
    
    if action == "removal":
        if is_fiedler:
            return 100.0 - raw_val if raw_val > 1.0 else (1.0 - raw_val) * 100.0
        elif raw_val <= 1.0:
            return raw_val * 100.0
        elif raw_val <= 50.0:
            return raw_val
        else:
            return 100.0 - raw_val
            
    else:  # Addition
        if is_fiedler:
            return raw_val - 100.0 if raw_val >= 100.0 else raw_val
        elif raw_val <= 2.0:
            return raw_val * 100.0
        elif raw_val > 50.0:
            return raw_val - 100.0
        else:
            return raw_val


def find_critical_tau_pct(df: pd.DataFrame, noise_col: str, metric_col: str, 
                          baseline_val: float, action: str, is_fiedler: bool = False) -> float:
    df_sorted = df.sort_values(noise_col)
    
    if is_fiedler:
        if baseline_val == 0 or pd.isnull(baseline_val):
            return np.nan
        # Relative Deviation: |val - base| / base > 0.50
        # Under Removal: val < 0.50 * baseline
        # Under Addition: val > 1.50 * baseline
        rel_deviation = (df_sorted[metric_col] - baseline_val).abs() / baseline_val
        failed_rows = df_sorted[rel_deviation > (1.0 - RETENTION_THRESHOLD)]
    else:
        # Standard Similarity Decay (ARI, GCC, Jaccard < 0.50 * baseline)
        target_val = baseline_val * RETENTION_THRESHOLD
        failed_rows = df_sorted[df_sorted[metric_col] < target_val]
        
    if not failed_rows.empty:
        raw_val = failed_rows.iloc[0][noise_col]
        return standardize_noise_pct(raw_val, action, is_fiedler)
    return np.nan


def extract_single_scenario(action: str, noise_type: str) -> pd.DataFrame:
    records = []
    
    # 1. GCC
    if PATHS["gcc"].exists():
        df = pd.read_csv(PATHS["gcc"])
        for raw_net, group in df.groupby("network"):
            net_name, model_type = parse_network(raw_net)
            if net_name == "Unknown" or (not INCLUDE_NULL_MODELS and model_type != "Empirical"): continue
            sub = group[group["action"].str.contains(action[:3], case=False, na=False) & 
                        group["noise_type"].str.contains(noise_type, case=False, na=False)]
            if not sub.empty:
                tau = find_critical_tau_pct(sub, "noise_level", "gcc_mean", 1.0, action)
                records.append({"Net": net_name, "Model": model_type, "Metric": "GCC", "Tau": tau})

    # 2. Fiedler (Fixed Divergence Logic)
    if PATHS["fiedler"].exists():
        df = pd.read_csv(PATHS["fiedler"])
        for raw_net, group in df.groupby("network_id"):
            net_name, model_type = parse_network(raw_net)
            if net_name == "Unknown" or (not INCLUDE_NULL_MODELS and model_type != "Empirical"): continue
            base_row = group[group["noise_type"] == "baseline"]
            baseline_val = base_row["avg_algebraic_connectivity"].values[0] if not base_row.empty else 0.0
            sub = group[group["edge_operation"].str.contains(action[:3], case=False, na=False) & 
                        group["noise_type"].str.contains(noise_type, case=False, na=False)]
            if not sub.empty:
                tau = find_critical_tau_pct(sub, "noise_level", "avg_algebraic_connectivity", baseline_val, action, is_fiedler=True)
                records.append({"Net": net_name, "Model": model_type, "Metric": "Fiedler (λ2)", "Tau": tau})

    # 3. Mesoscale Algorithms
    meso_algos = [("infomap", "Meso_Infomap"), ("louvain", "Meso_Louvain"), 
                  ("leiden", "Meso_Leiden"), ("label_propagation", "Meso_LabelProp")]
    for key, algo_label in meso_algos:
        if PATHS[key].exists():
            df = pd.read_csv(PATHS[key])
            for raw_net, group in df.groupby("base_network_name"):
                net_name, model_type = parse_network(raw_net)
                if net_name == "Unknown" or (not INCLUDE_NULL_MODELS and model_type != "Empirical"): continue
                sub = group[group["type_of_noise"].str.contains(action[:3], case=False, na=False) & 
                            group["target"].str.contains(noise_type, case=False, na=False)]
                if not sub.empty:
                    tau = find_critical_tau_pct(sub, "level", "vs_baseline_ari_mean", 1.0, action)
                    records.append({"Net": net_name, "Model": model_type, "Metric": algo_label, "Tau": tau})

    # 4. Local Seed Expansion (Jaccard k=25)
    if PATHS["seed_expansion"].exists():
        df = pd.read_csv(PATHS["seed_expansion"])
        df["k_category_str"] = df["k_category"].astype(str)
        df = df[df["k_category_str"].str.startswith(str(LOCAL_K_TARGET))]
        for (raw_net, algo), group in df.groupby(["network", "algorithm"]):
            net_name, model_type = parse_network(raw_net)
            if net_name == "Unknown" or (not INCLUDE_NULL_MODELS and model_type != "Empirical"): continue
            sub = group[group["modification_type"].str.contains(action[:3], case=False, na=False) & 
                        group["perturbation_type"].str.contains(noise_type, case=False, na=False)]
            if not sub.empty:
                df_sorted = sub.sort_values("noise_level")
                baseline_val = df_sorted.iloc[0]["jaccard_mean"]
                tau = find_critical_tau_pct(sub, "noise_level", "jaccard_mean", baseline_val, action)
                records.append({"Net": net_name, "Model": model_type, "Metric": f"Local_{algo.upper()}", "Tau": tau})

    df_scen = pd.DataFrame(records)
    if df_scen.empty: return pd.DataFrame()
    df_scen["Row_ID"] = df_scen["Net"] if not INCLUDE_NULL_MODELS else df_scen["Net"] + " (" + df_scen["Model"] + ")"
    return df_scen.pivot_table(index="Row_ID", columns="Metric", values="Tau", aggfunc="min")


# ==============================================================================
# MAIN PLOTTING PIPELINE
# ==============================================================================
def main():
    plt.rcParams.update({
        'font.size': 13,
        'font.weight': 'bold',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 14,
        'figure.titlesize': 20
    })

    actions = ["removal", "addition"]
    noises  = ["hub", "periphery", "random"]
    
    col_order = [
        "GCC", "Fiedler (λ2)",
        "Meso_Infomap", "Meso_Louvain", "Meso_Leiden", "Meso_LabelProp",
        "Local_DIAMOND", "Local_FIRST_NEIGHBORS", "Local_RWR_ROW", "Local_RWR_SYM"
    ]
    col_labels = [
        "GCC", "Fiedler",
        "Infomap", "Louvain", "Leiden", "LabelProp",
        "DIAMOnD", "1st Nbrs", "RWR Row", "RWR Sym"
    ]

    fig, axes = plt.subplots(3, 2, figsize=(22, 14), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.10, hspace=0.28, bottom=0.12)

    for r_idx, n_type in enumerate(noises):
        for c_idx, act in enumerate(actions):
            ax = axes[r_idx, c_idx]
            
            matrix = extract_single_scenario(act, n_type)
            if matrix.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            matrix = matrix.reindex(columns=col_order)
            
            v_max = MAX_TESTED_REMOVAL if act == "removal" else MAX_TESTED_ADDITION
            matrix_plot = matrix.clip(upper=v_max).fillna(v_max)
            
            def format_annotation(val):
                if pd.isnull(val) or val >= v_max:
                    return f">{v_max:.0f}%"
                prefix = "+" if act == "addition" else "-"
                return f"{prefix}{val:.0f}%"

            annot_array = matrix.map(format_annotation).values
            
            sns.heatmap(
                matrix_plot, 
                ax=ax, 
                annot=annot_array, 
                fmt="", 
                cmap="YlOrRd_r",
                vmin=0.0, 
                vmax=v_max,
                cbar=False,
                linewidths=1.2,
                linecolor="white",
                annot_kws={"size": 12, "weight": "bold"}
            )
            
            ax.set_title(f"{n_type.upper()} EDGE {act.upper()}", pad=10)
            ax.set_xticklabels(col_labels, rotation=40, ha='right', weight='bold')
            ax.set_ylabel("")
            ax.set_xlabel("")

    # Vertical dividers for scale boundaries
    for ax in axes.flat:
        ax.axvline(2, color='black', linewidth=2.5)  # Global | Meso
        ax.axvline(6, color='black', linewidth=2.5)  # Meso | Local

    fig.suptitle("Cross-Scale Failure Threshold Overview (τ_50% Retention/Divergence)", y=0.98, weight='bold')

    # DEDICATED INDIVIDUAL COLORBARS
    cax_rem = fig.add_axes([0.18, 0.02, 0.28, 0.022])
    sm_rem = plt.cm.ScalarMappable(cmap="YlOrRd_r", norm=plt.Normalize(vmin=0.0, vmax=MAX_TESTED_REMOVAL))
    sm_rem.set_array([])
    cb_rem = fig.colorbar(sm_rem, cax=cax_rem, orientation="horizontal")
    cb_rem.set_label("Edge Removal Noise Scale (% Edges Removed)", weight='bold', fontsize=12)
    cb_rem.set_ticks([0, 25, 50])
    cb_rem.set_ticklabels(["0% (Ultra-Fragile)", "25%", ">50% (Max Noise Survived)"])

    cax_add = fig.add_axes([0.56, 0.02, 0.28, 0.022])
    sm_add = plt.cm.ScalarMappable(cmap="YlOrRd_r", norm=plt.Normalize(vmin=0.0, vmax=MAX_TESTED_ADDITION))
    sm_add.set_array([])
    cb_add = fig.colorbar(sm_add, cax=cax_add, orientation="horizontal")
    cb_add.set_label("Edge Addition Noise Scale (% Edges Added)", weight='bold', fontsize=12)
    cb_add.set_ticks([0, 100, 200])
    cb_add.set_ticklabels(["+0% (Ultra-Fragile)", "+100%", ">200% (Max Noise Survived)"])

    out_png = Path(f"master_cross_scale_summary_fixed_fiedler_{'all' if INCLUDE_NULL_MODELS else 'empirical'}.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"\n✅ Updated plot (Fixed Fiedler + Jaccard k=25) saved to: {out_png.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()