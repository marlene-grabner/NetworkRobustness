"""
Topological properties of the null model ensembles (ER, configuration model
and microcanonical SBM, N_REPLICATES rewired replicates each, see
er_and_conf/ and sbm/).

Every replicate gets the same profile as the empirical networks
(TopologicalProperties.get_network_profile). Per network and model the mean
and the standard deviation over the replicates are reported.

Writes:
  outputs/models/null_model_ensemble_properties_per_rep.csv   every replicate
  outputs/models/null_model_ensemble_properties.csv           mean and SD
  outputs/latex_tables/models/topological_properties/
      null_model_ensemble_properties.tex     one table per null model type
      null_model_ensemble_connectedness.tex  nodes, components, GCC

Run from the repository root (takes hours, mostly the exact average path
length of the ppi and astro replicates):
  uv run python scripts/models/properties_null_model_ensembles.py
Rebuild only the tables from the saved per-replicate csv:
  uv run python scripts/models/properties_null_model_ensembles.py --tables-only
"""

import os
import sys
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from NoiseEffect import TopologicalProperties
from NoiseEffect.utils import df_to_latex, save_latex_tables
from NoiseEffect.utils.write_latex_table_from_df import _fmt_number

NULL_DIR = "./data/baseline_networks/null_models/"
N_REPLICATES = 10
N_PROCESSES = 8

NETWORKS = {
    "ppi": "PPI",
    "power": "Power Grid",
    "astro": "Astrophysics",
    "wiki": "Wikipedia",
    "huri": "HuRI",
}
MODELS = {"er": "ER", "conf": "Conf", "sbm": "SBM"}

CSV_DIR = "./outputs/models/"
TEX_DIR = "outputs/latex_tables/models/topological_properties/"

# Rows that count something: shown without decimals unless they vary
COUNT_ROWS = {
    "Nodes_Full",
    "Edges_Full",
    "N_Components",
    "N_Singletons",
    "N_NonGCC_Nodes",
    "GCC_Size",
    "Nodes",
    "Edges",
}


def load_null(path):
    """Load a null model edgelist plus its isolated-node sidecar, if any."""
    G = nx.read_edgelist(path, delimiter=",")
    sidecar = path.rsplit(".", 1)[0] + "_isolated_nodes.csv"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            G.add_nodes_from(n.strip() for n in f.read().split(",") if n.strip())
    return G


def profile_replicate(task):
    model, network, rep = task
    G = load_null(f"{NULL_DIR}{model}_ensemble/{network}_{model}_rep{rep}.csv")
    profile = TopologicalProperties.get_network_profile(G)
    print(f"  {network}_{model} rep {rep} done", flush=True)
    return {"model": model, "network": network, "rep": rep, **profile}


def mean_pm_sd(values, count):
    """'mean +- SD' over replicates; a single value if all replicates agree."""
    values = np.asarray(values, dtype=float)
    if np.all(values == values[0]):
        x = values[0]
        return f"{int(x)}" if count else _fmt_number(x, 3, 1e-3, 1e5)
    mean, sd = values.mean(), np.std(values, ddof=1)
    if count:
        return f"{mean:.1f} $\\pm$ {sd:.1f}"
    scale = max(abs(mean), sd)
    if scale < 1e-3:
        # small values share one exponent: (a +- b) x 10^e
        exp = int(np.floor(np.log10(scale)))
        return f"$({mean / 10**exp:.2f} \\pm {sd / 10**exp:.2f}) \\times 10^{{{exp}}}$"
    sd_text = f"{sd:.3f}" if sd >= 5e-4 else "$<$0.001"
    mean = round(mean, 3) + 0.0  # no "-0.000"
    return f"{mean:.3f} $\\pm$ {sd_text}"


def metrics_scope(values):
    n_gcc = sum(v == "GCC" for v in values)
    if n_gcc == 0:
        return "full"
    if n_gcc == len(values):
        return "GCC"
    return f"GCC ({n_gcc}/{len(values)})"


def summary_cell(per_rep, model, network, metric):
    values = per_rep[(per_rep.model == model) & (per_rep.network == network)][metric]
    if metric == "Metrics_Scope":
        return metrics_scope(list(values))
    return mean_pm_sd(values, metric in COUNT_ROWS)


if __name__ == "__main__":
    per_rep_path = CSV_DIR + "null_model_ensemble_properties_per_rep.csv"
    if "--tables-only" in sys.argv:
        # rebuild the tables from the saved profiles (the path length on ppi
        # and astro makes the full run take hours)
        per_rep = pd.read_csv(per_rep_path)
    else:
        tasks = [
            (model, network, rep)
            for model in MODELS
            for network in NETWORKS
            for rep in range(N_REPLICATES)
        ]
        with Pool(N_PROCESSES) as pool:
            per_rep = pd.DataFrame(pool.map(profile_replicate, tasks, chunksize=1))
        per_rep.to_csv(per_rep_path, index=False)

    metrics = [c for c in per_rep.columns if c not in ("model", "network", "rep")]
    numeric = [m for m in metrics if m != "Metrics_Scope"]
    agg = per_rep.groupby(["model", "network"])[numeric].agg(["mean", "std"])
    agg.to_csv(CSV_DIR + "null_model_ensemble_properties.csv")

    # -------- One properties table per null model type --------
    tables = []
    for model, model_label in MODELS.items():
        df = pd.DataFrame(
            {
                label: [summary_cell(per_rep, model, network, m) for m in metrics]
                for network, label in NETWORKS.items()
            },
            index=[m.replace("_", " ") for m in metrics],
        )
        tables.append(
            df_to_latex(
                df=df,
                caption=f"Topological properties of the {model_label} null models, mean "
                f"$\\pm$ SD over {N_REPLICATES} replicates (a single value means all "
                "replicates agree). Metrics Scope: whether the structural metrics are "
                "taken on the GCC (in how many replicates) or on the full graph.",
                label=f"tab:null_model_ensemble_properties_{model}",
                escape_text=False,
            )
        )
    save_latex_tables(TEX_DIR + "null_model_ensemble_properties.tex", tables)

    # -------- Connectedness of all null models --------
    rows = {}
    for model in MODELS:
        for network in NETWORKS:
            rows[f"{NETWORKS[network]} ({MODELS[model]})"] = {
                "Nodes": summary_cell(per_rep, model, network, "Nodes_Full"),
                "Components": summary_cell(per_rep, model, network, "N_Components"),
                "Singletons": summary_cell(per_rep, model, network, "N_Singletons"),
                "GCC Size": summary_cell(per_rep, model, network, "GCC_Size"),
                "GCC Fraction": summary_cell(per_rep, model, network, "GCC_Fraction"),
            }
    connectedness = pd.DataFrame.from_dict(rows, orient="index")
    save_latex_tables(
        TEX_DIR + "null_model_ensemble_connectedness.tex",
        df_to_latex(
            df=connectedness,
            caption=f"Connectedness of the null models, mean $\\pm$ SD over "
            f"{N_REPLICATES} replicates (a single value means all replicates agree). "
            "Nodes includes isolated nodes (singletons). All empirical networks are "
            "connected.",
            label="tab:connectedness_null_models",
            escape_text=False,
            column_format="lrrrrr",
        ),
    )
