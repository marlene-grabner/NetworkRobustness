# %%
import networkx as nx
import random
from NoiseEffect.CommunityDetection import benchmarkBaselineStabilityAlgorithm
from NoiseEffect.CommunityDetection.Visualisations import plotStabilityResults, plotSpreadOfStabilityResults
from NoiseEffect.CommunityDetection.utils import load_isolated_nodes, build_full_graph
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
# -------------------------------------------------------------------------
# 1. Loading the base networks
# -------------------------------------------------------------------------

#%%
def load_graph(csv_path):
    """
    Loads an edgelist into a complete igraph.Graph, including any degree-0 nodes
    recorded in the network's isolated-nodes sidecar CSV (only null models
    generated via ER/SBM can have these, since an edgelist can't represent a
    node with no edges).
    """
    df = pd.read_csv(csv_path, sep=',', names=['source', 'target'], dtype=str)
    isolated_nodes = load_isolated_nodes(csv_path)
    g, _node_order, _has_edge = build_full_graph(df, isolated_nodes)
    return g

############################################
# Protein Interaction Network & Null Models

G_ig_ppi = load_graph("../../../data/baseline_networks/ppi.csv")
G_ig_ppi_er = load_graph("../../../data/baseline_networks/null_models/ppi_er.csv")
G_ig_ppi_config = load_graph("../../../data/baseline_networks/null_models/ppi_conf.csv")
G_ig_ppi_sbm = load_graph("../../../data/baseline_networks/null_models/ppi_sbm.csv")

############################################
# Western US Power Grid Network & Null Models

G_ig_power = load_graph("../../../data/baseline_networks/power.csv")
G_ig_power_er = load_graph("../../../data/baseline_networks/null_models/power_er.csv")
G_ig_power_config = load_graph("../../../data/baseline_networks/null_models/power_conf.csv")
G_ig_power_sbm = load_graph("../../../data/baseline_networks/null_models/power_sbm.csv")

############################################
# Astrophysics Collaboration Network & Null Models

G_ig_astro = load_graph("../../../data/baseline_networks/astro.csv")
G_ig_astro_er = load_graph("../../../data/baseline_networks/null_models/astro_er.csv")
G_ig_astro_config = load_graph("../../../data/baseline_networks/null_models/astro_conf.csv")
G_ig_astro_sbm = load_graph("../../../data/baseline_networks/null_models/astro_sbm.csv")

############################################
# Wikipedia Vote Network & Null Models

G_ig_wiki = load_graph("../../../data/baseline_networks/wiki.csv")
G_ig_wiki_er = load_graph("../../../data/baseline_networks/null_models/wiki_er.csv")
G_ig_wiki_config = load_graph("../../../data/baseline_networks/null_models/wiki_conf.csv")
G_ig_wiki_sbm = load_graph("../../../data/baseline_networks/null_models/wiki_sbm.csv")

# -------------------------------------------------------------------------
# 2. Graphs to benchmark
# -------------------------------------------------------------------------


graphs_to_benchmark = {
    "Protein Interaction": G_ig_ppi,
    "Protein Interaction (ER)": G_ig_ppi_er,
    "Protein Interaction (Config)": G_ig_ppi_config,
    "Protein Interaction (SBM)": G_ig_ppi_sbm,
    "Western US Power Grid": G_ig_power,
    "Western US Power Grid (ER)": G_ig_power_er,
    "Western US Power Grid (Config)": G_ig_power_config,
    "Western US Power Grid (SBM)": G_ig_power_sbm,
    "Astrophysics Collaboration": G_ig_astro,
    "Astrophysics Collaboration (ER)": G_ig_astro_er,
    "Astrophysics Collaboration (Config)": G_ig_astro_config,
    "Astrophysics Collaboration (SBM)": G_ig_astro_sbm,
    "Wikipedia Vote": G_ig_wiki,
    "Wikipedia Vote (ER)": G_ig_wiki_er,
    "Wikipedia Vote (Config)": G_ig_wiki_config,
    "Wikipedia Vote (SBM)": G_ig_wiki_sbm,
}


# -------------------------------------------------------------------------
# 2. Create random seeds to test with
# -------------------------------------------------------------------------

#### SPECIFY NUMBER OF SEEDS ####

num_of_seeds = 10
seeds = random.sample(range(1, 1000), num_of_seeds)

# -------------------------------------------------------------------------
# 3. Analyse the stability of the different algorithms on the test networks
# -------------------------------------------------------------------------
#

results_leiden_n_iterations_default = {}
results_infomap_n_iterations_default = {}
results_louvaib_n_iterations_default = {}
results_label_propagation_n_iterations_default = {}
import time
start = time.time()
for name, G in tqdm(graphs_to_benchmark.items()):
    results_leiden_n_iterations_default[name] = benchmarkBaselineStabilityAlgorithm(
        G, seeds, algorithm="leiden"
    )
    results_infomap_n_iterations_default[name] = benchmarkBaselineStabilityAlgorithm(
        G, seeds, algorithm="infomap"
    )
    results_louvaib_n_iterations_default[name] = benchmarkBaselineStabilityAlgorithm(
        G, seeds, algorithm="louvain"
    )
    results_label_propagation_n_iterations_default[name] = benchmarkBaselineStabilityAlgorithm(
        G, seeds, algorithm="label_propagation"
    )
print(f"Time taken for benchmarking: {time.time() - start:.2f} seconds")


# %%
# -------------------------------------------------------------------------
# 4. Visualize the results
# -------------------------------------------------------------------------
#


plotStabilityResults(
    results_leiden_n_iterations_default,
    algorithm_name="Leiden Algorithm",
    extra_info="n_iterations = 2, n = 45",
    #save_path="./new_network_visualisations/leiden_default_iterations_stability.pdf",
)

plotStabilityResults(
    results_infomap_n_iterations_default,
    algorithm_name="Infomap Algorithm",
    extra_info="n_iterations = 2, n = 45",
    #save_path="./new_network_visualisations/infomap_default_iterations_stability.pdf",
)

plotStabilityResults(
    results_louvaib_n_iterations_default,
    algorithm_name="Louvain Algorithm",
    extra_info="n_iterations = default, n = 45",
    #save_path="./new_network_visualisations/louvain_default_iterations_stability.pdf",
)

plotStabilityResults(
    results_label_propagation_n_iterations_default,
    algorithm_name="Label Propagation Algorithm",
    extra_info="n_iterations = default, n = 45",
    #save_path="./new_network_visualisations/label_propagation_default_iterations_stability.pdf",
)


# Spread of stability results

plotSpreadOfStabilityResults(
    results_leiden_n_iterations_default,
    measurement="ari",
    title="Leiden Algorithm Stability (ARI)",
)

plotSpreadOfStabilityResults(
    results_infomap_n_iterations_default,
    measurement="ari",
    title="Infomap Algorithm Stability (ARI)",
)

plotSpreadOfStabilityResults(
    results_louvaib_n_iterations_default,
    measurement="ari",
    title="Louvain Algorithm Stability (ARI)",
)

plotSpreadOfStabilityResults(
    results_label_propagation_n_iterations_default,
    measurement="ari",
    title="Label Propagation Algorithm Stability (ARI)",
)

# %%
print(f"results_leiden_n_iterations_default: {results_leiden_n_iterations_default}")
print(f"results_infomap_n_iterations_default: {results_infomap_n_iterations_default}")
print(f"results_louvaib_n_iterations_default: {results_louvaib_n_iterations_default}")
print(f"results_label_propagation_n_iterations_default: {results_label_propagation_n_iterations_default}")
# %%
for name, results in results_leiden_n_iterations_default.items():
    print(f"Leiden Algorithm - {name}:")
    aris = []
    for result in results:
        aris.append(result["ari"])
    print(f"ARI values: {np.mean(aris):.4f} ± {np.std(aris):.4f}")
        
# %%
