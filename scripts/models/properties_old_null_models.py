"""
Topological properties of the OLD null models (one network each), kept for
the appendix "Old null models":
  ER    networkx gnm_random_graph
  Conf  igraph Viger-Latapy (simple and forced connected)
  SBM   canonical degree-corrected SBM from generate_sbm (multi-edges and
        self-loops collapsed, so fewer edges and isolated nodes)
They were replaced by the rewired 10-replicate ensembles, see
properties_null_model_ensembles.py.

Run from the repository root:
  uv run python scripts/models/properties_old_null_models.py
"""

import os

import networkx as nx
import pandas as pd
from NoiseEffect import TopologicalProperties
from NoiseEffect.utils import df_to_latex, save_latex_tables

OLD_NULL_DIR = "./data/sbm_models_not_microcanoical_null_models_once/"

NETWORKS = {
    "ppi": "PPI",
    "power": "Power Grid",
    "astro": "Astrophysics",
    "wiki": "Wikipedia",
    "huri": "HuRI",
}
MODELS = {"er": "ER", "conf": "Conf", "sbm": "SBM"}


def load_null(path):
    """Load a null model edgelist plus its isolated-node sidecar, if any."""
    G = nx.read_edgelist(path, delimiter=",")
    sidecar = path.rsplit(".", 1)[0] + "_isolated_nodes.csv"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            G.add_nodes_from(n.strip() for n in f.read().split(",") if n.strip())
    return G


profiles = {}
for model, model_label in MODELS.items():
    for network, network_label in NETWORKS.items():
        G = load_null(f"{OLD_NULL_DIR}{network}_{model}.csv")
        profiles[f"{network_label} ({model_label})"] = (
            TopologicalProperties.get_network_profile(G)
        )
        print(f"{network}_{model} done", flush=True)

df = pd.DataFrame(profiles)
df.to_csv("./outputs/models/old_null_model_network_properties.csv", index=True, header=True)

save_latex_tables(
    file_path="outputs/latex_tables/models/topological_properties/old_null_model_network_properties.tex",
    tables=df_to_latex(
        df=df,
        caption="Topological properties of the old null models (one network each: "
        "networkx ER, Viger--Latapy configuration model and canonical SBM).",
        label="tab:old_null_model_network_properties",
    ),
)
