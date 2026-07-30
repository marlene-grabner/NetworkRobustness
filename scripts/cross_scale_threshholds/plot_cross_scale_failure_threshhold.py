from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RETENTION_THRESHOLD = 0.50  # 50% baseline retention threshold

LOCAL_K_TARGET = "50"

# Set True to generate plot for ALL 16 models (Empirical + Null Models)
# Set False to generate plot for the 4 EMPIRICAL networks only
INCLUDE_NULL_MODELS = False

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
# UNIT NORMALIZATION HELPER
# ==============================================================================
def parse_network(raw_str: str) -> tuple[str, str]:
    s = str(raw_str).lower().strip()
    for raw_key, (net_name, model_type) in NETWORK_MAP.items():
        if raw_key == s or raw_key in s:
            return net_name, model_type
    return "Unknown", "Unknown"


def standardize_noise_pct(raw_val: float, action: str, is_fiedler: bool = False) -> float:
    """
    Standardizes all raw CSV noise levels to % Edges Modified (ΔE / E_0 * 100%).
    - Removal: Output range 0% to 50%
    - Addition: Output range 0% to 100% (+100% added edges = doubling baseline graph)
    """
    if pd.isnull(raw_val):
        return np.nan
        
    raw_val = float(raw_val)
    
    if action == "removal":
        if is_fiedler:
            # Fiedler remaining size: 100.0 -> 50.0  ==> Noise % = 100.0 - raw_val
            return 100.0 - raw_val
        elif raw_val <= 1.0:
            # Fraction removed: 0.05 -> 0.50  ==> Noise % = raw_val * 100
            return raw_val * 100.0
        else:
            return raw_val
            
    else:  # Addition
        if is_fiedler or raw_val > 50.0:
            # Relative network size: 105.0 -> 200.0 ==> Added Edges % = raw_val - 100.0
            return raw_val - 100.0
        elif raw_val <= 2.0:
            # Fraction added: 0.05 -> 1.0 ==> Added Edges % = raw_val * 100
            return raw_val * 100.0
        else:
            return raw_val


def find_critical_tau_pct(df: pd.DataFrame, noise_col: str, metric_col: str, 
                          baseline_val: float, action: str, is_fiedler: bool = False) -> float:
    df_sorted = df.sort_values(noise_col)
    target_val = baseline_val * RETENTION_THRESHOLD
    
    failed_rows = df_sorted[df_sorted[metric_col] < target_val]
    if not failed_rows.empty:
        raw_val = failed_rows.iloc[0][noise_col]
        return standardize_noise_pct(raw_val, action, is_fiedler)
            
    return np.nan # Did not fail within tested boundary


# ==============================================================================
# DATA EXTRACTOR
# ==============================================================================
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

    # 2. Fiedler
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

    # 4. Local Seed Expansion
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
                baseline_val = df_sorted.iloc[0]["auprc_mean"]
                tau = find_critical_tau_pct(sub, "noise_level", "auprc_mean", baseline_val, action)
                records.append({"Net": net_name, "Model": model_type, "Metric": f"Local_{algo.upper()}", "Tau": tau})

    df_scen = pd.DataFrame(records)
    if df_scen.empty: return pd.DataFrame()
    
    df_scen["Row_ID"] = df_scen["Net"] if not INCLUDE_NULL_MODELS else df_scen["Net"] + " (" + df_scen["Model"] + ")"
    return df_scen.pivot_table(index="Row_ID", columns="Metric", values="Tau", aggfunc="min")


# ==============================================================================
# PLOTTING PIPELINE
# ==============================================================================
def main():
    # Set global matplotlib parameters for publication-quality readability
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
    plt.subplots_adjust(wspace=0.10, hspace=0.28)

    for r_idx, n_type in enumerate(noises):
        for c_idx, act in enumerate(actions):
            ax = axes[r_idx, c_idx]
            
            matrix = extract_single_scenario(act, n_type)
            if matrix.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
                
            matrix = matrix.reindex(columns=col_order)
            
            # SYMMETRIC COLOR NORMALIZATION:
            # Removal max boundary = 50.0% (Halving edges)
            # Addition max boundary = 100.0% (Doubling edges)
            max_tested_noise = 50.0 if act == "removal" else 100.0
            
            # Normalize matrix to [0.0, 1.0] for color mapping
            matrix_norm = matrix / max_tested_noise
            # Cap values >100% addition at 1.0 color intensity
            matrix_norm_plot = matrix_norm.clip(upper=1.0).fillna(1.0)
            
            # Human-readable cell annotations
            def format_annotation(val):
                if pd.isnull(val) or val > max_tested_noise:
                    return ">50%" if act == "removal" else ">100%"
                prefix = "+" if act == "addition" else "-"
                return f"{prefix}{val:.0f}%"

            annot_array = matrix.map(format_annotation).values
            
            # Plot Heatmap with shared normalized scale [0.0, 1.0]
            sns.heatmap(
                matrix_norm_plot, 
                ax=ax, 
                annot=annot_array, 
                fmt="", 
                cmap="YlOrRd_r",  # Dark Red = Fragile, Light Yellow = Robust
                vmin=0.0, 
                vmax=1.0,
                cbar=False,
                linewidths=1.2,
                linecolor="white",
                annot_kws={"size": 12, "weight": "bold"}
            )
            
            ax.set_title(f"{n_type.upper()} EDGE {act.upper()}", pad=10)
            ax.set_xticklabels(col_labels, rotation=40, ha='right', weight='bold')
            ax.set_ylabel("")
            ax.set_xlabel("")

    # Add Scale Partition Vertical Dividers
    for ax in axes.flat:
        ax.axvline(2, color='black', linewidth=2.5)  # Global | Meso
        ax.axvline(6, color='black', linewidth=2.5)  # Meso | Local

    # Global Main Title
    fig.suptitle("Cross-Scale Failure Threshold Overview (τ_50% Retention)", y=0.98, weight='bold')

    # Add Colorbar Legend at Bottom
    cbar_ax = fig.add_axes([0.25, 0.02, 0.50, 0.025])
    sm = plt.cm.ScalarMappable(cmap="YlOrRd_r", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("2-Fold Density Perturbed Scale (0.0 = Immediate Failure  ➜  1.0 = 2-Fold Boundary Survived)", weight='bold', fontsize=13)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0% (Ultra-Fragile)", "1-Fold Perturbation", "2-Fold Boundary (>50% Rem / >100% Add)"])

    out_png = Path(f"master_cross_scale_summary_{'all' if INCLUDE_NULL_MODELS else 'empirical'}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"\n✅ Prettier symmetric master summary saved to: {out_png.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()