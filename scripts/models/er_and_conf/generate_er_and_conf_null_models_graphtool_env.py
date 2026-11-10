#!/usr/bin/env python3
"""
Erdos-Renyi and configuration model null models, made the same way as the SBM
ensemble (sbm/generate_sbm_null_models_graphtool_env.py): the empirical
network is rewired with graph-tool's random_rewire, and N_REPLICATES
independent draws are made per network. Every replicate starts from the
empirical network.

  ER            model="erdos"          G(N, E): node and edge count kept,
                                       edges placed uniformly at random.
  configuration model="configuration"  every node degree kept exactly.

Both are simple graphs (no self-loops, no multi-edges). Unlike the old igraph
VL generator, the configuration model is NOT forced to be connected, so it can
have small components (e.g. two degree-1 nodes linked to each other). It
cannot have isolated nodes, because every degree is kept.

The ER model can isolate nodes. A headerless edgelist drops them, so they are
written to a sidecar: {network}_er_rep{r}_isolated_nodes.csv (comma-separated
node ids, only written when there are any).

Writes, refusing to overwrite:
  data/baseline_networks/null_models/er_ensemble/{network}_er_rep{r}.csv
  data/baseline_networks/null_models/conf_ensemble/{network}_conf_rep{r}.csv

# ====================================================================
# DISCLAIMER: requires the graph-tool package (use graphtool-env).
# Run from the repository root:
#   conda run -n graphtool-env python scripts/models/er_and_conf/generate_er_and_conf_null_models_graphtool_env.py
# ====================================================================
"""

import os
import sys

import graph_tool.all as gt
import networkx as nx
import numpy as np

# Same network list and loader as the SBMs (incl. removing astro's 59
# self-loops), so all three null model types start from the same graph.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sbm"))
from sbm_common import BASELINE_DIR, NETWORKS, load_baseline, to_graph_tool  # noqa: E402

NULL_DIR = BASELINE_DIR + "null_models/"
MODELS = {
    # suffix: (random_rewire model, output folder)
    "er": ("erdos", NULL_DIR + "er_ensemble/"),
    "conf": ("configuration", NULL_DIR + "conf_ensemble/"),
}

N_REPLICATES = 10
N_REWIRE_ITER = 100


def out_path(suffix, name, rep):
    return f"{MODELS[suffix][1]}{name}_{suffix}_rep{rep}.csv"


def sidecar_path(edge_path):
    return edge_path.rsplit(".", 1)[0] + "_isolated_nodes.csv"


def generate_null_edgelist(g, model, idx_to_node, original_nodes, n_iter=100):
    """Rewire a copy of g with random_rewire and return it as a networkx graph."""
    g_null = g.copy()
    gt.random_rewire(
        g_null, model=model, parallel_edges=False, self_loops=False, n_iter=n_iter
    )

    # --- verify the constraints actually held ---
    assert g_null.num_vertices() == g.num_vertices(), "node count changed"
    assert g_null.num_edges() == g.num_edges(), "edge count changed"
    if model == "configuration":
        assert np.array_equal(
            g.degree_property_map("total").a, g_null.degree_property_map("total").a
        ), "degrees changed"

    null_nx = nx.Graph()
    null_nx.add_nodes_from(original_nodes)  # keeps nodes ER may isolate
    null_nx.add_edges_from(
        (idx_to_node[int(e.source())], idx_to_node[int(e.target())])
        for e in g_null.edges()
    )
    assert null_nx.number_of_edges() == g.num_edges(), "multi-edges were collapsed"
    assert nx.number_of_selfloops(null_nx) == 0, "self-loops present"
    return null_nx


def write_null(null_nx, edge_path):
    nx.write_edgelist(null_nx, edge_path, delimiter=",", data=False)
    isolated = [str(n) for n, d in null_nx.degree() if d == 0]
    if isolated:
        with open(sidecar_path(edge_path), "w") as f:
            f.write(",".join(isolated))
    return len(isolated)


def edge_jaccard(G, H):
    e_g = {frozenset(e) for e in G.edges()}
    e_h = {frozenset(e) for e in H.edges()}
    return len(e_g & e_h) / len(e_g | e_h)


if __name__ == "__main__":
    # Check before doing any work, so nothing is overwritten by accident
    for suffix in MODELS:
        for name in NETWORKS:
            for rep in range(N_REPLICATES):
                for path in (out_path(suffix, name, rep), sidecar_path(out_path(suffix, name, rep))):
                    if os.path.exists(path):
                        raise FileExistsError(f"{path} already exists, not overwriting it.")
    for _, folder in MODELS.values():
        os.makedirs(folder, exist_ok=True)

    # One thread, so that a seed always gives the same replicate
    gt.openmp_set_num_threads(1)

    for name in NETWORKS:
        G = load_baseline(name)
        print(f"\n=== {name} (N={G.number_of_nodes()}, E={G.number_of_edges()}) ===")
        original_nodes = list(G.nodes())
        g, _, idx_to_node = to_graph_tool(G)

        for suffix, (model, _) in MODELS.items():
            for rep in range(N_REPLICATES):
                # seed_rng(0) means "seed from system entropy" in graph-tool,
                # which is not reproducible, so seeds start at 1
                gt.seed_rng(rep + 1)
                np.random.seed(rep + 1)
                null_nx = generate_null_edgelist(
                    g, model, idx_to_node, original_nodes, n_iter=N_REWIRE_ITER
                )
                n_isolated = write_null(null_nx, out_path(suffix, name, rep))
                gcc = max(len(c) for c in nx.connected_components(null_nx))
                print(
                    f"  {suffix:4s} rep {rep}: E={null_nx.number_of_edges()}, "
                    f"jaccard={edge_jaccard(G, null_nx):.4f}, "
                    f"GCC={gcc / null_nx.number_of_nodes():.4f}, "
                    f"components={nx.number_connected_components(null_nx)}, "
                    f"isolated={n_isolated}",
                    flush=True,
                )
