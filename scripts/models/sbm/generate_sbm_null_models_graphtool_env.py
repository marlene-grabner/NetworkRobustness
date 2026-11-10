#!/usr/bin/env python3
"""
Step 2 of 2: generate the SBM null models from the fitted blocks
(fit_sbm_blocks_graphtool_env.py).

Each null model is a microcanonical SBM: the empirical network rewired so that
every node degree and the edge count of every block pair stay exact. An
ensemble of N_REPLICATES is made per network. Every replicate starts from the
empirical network, so replicates are independent draws of the same blocks.

Rewiring cannot isolate nodes, so no *_isolated_nodes.csv sidecar is needed.

Writes, refusing to overwrite:
  data/baseline_networks/null_models/sbm_ensemble/{network}_sbm_rep{r}.csv

# ====================================================================
# DISCLAIMER: requires the graph-tool package (use graphtool-env).
# Run from the repository root:
#   conda run -n graphtool-env python scripts/models/sbm/generate_sbm_null_models_graphtool_env.py
# ====================================================================
"""

import json
import os

import graph_tool.all as gt
import networkx as nx
import numpy as np

from sbm_common import BASELINE_DIR, META_DIR, NETWORKS, load_baseline, to_graph_tool

OUT_DIR = BASELINE_DIR + "null_models/sbm_ensemble/"

N_REPLICATES = 10
N_REWIRE_ITER = 100


def generate_null_edgelist(g, b_array, idx_to_node, original_nodes, n_iter=100):
    """Microcanonical DC-SBM null via degree- and block-preserving rewiring."""
    g_null = g.copy()
    b_null = g_null.new_vertex_property("int", vals=b_array)  # same block labels
    b_orig = g.new_vertex_property("int", vals=b_array)

    gt.random_rewire(
        g_null,
        model="constrained-configuration",
        block_membership=b_null,
        parallel_edges=False,
        self_loops=False,
        n_iter=n_iter,
    )

    # --- verify the constraints actually held ---
    assert np.array_equal(
        g.degree_property_map("total").a, g_null.degree_property_map("total").a
    ), "degrees changed"
    # compared as sparse matrices: block labels go up to N, so dense ones need GBs
    e_orig = gt.BlockState(g, b=b_orig).get_matrix()
    e_null = gt.BlockState(g_null, b=b_null).get_matrix()
    assert (e_orig != e_null).nnz == 0, "block edge counts changed"

    null_nx = nx.Graph()
    null_nx.add_nodes_from(original_nodes)
    null_nx.add_edges_from(
        (idx_to_node[int(e.source())], idx_to_node[int(e.target())])
        for e in g_null.edges()
    )
    return null_nx


if __name__ == "__main__":
    # Check before doing any work, so nothing is overwritten by accident
    for name in NETWORKS:
        for rep in range(N_REPLICATES):
            if os.path.exists(f"{OUT_DIR}{name}_sbm_rep{rep}.csv"):
                raise FileExistsError(
                    f"{OUT_DIR}{name}_sbm_rep{rep}.csv already exists, not overwriting it."
                )
    os.makedirs(OUT_DIR, exist_ok=True)

    # One thread, so that a seed always gives the same replicate
    gt.openmp_set_num_threads(1)

    for name, meta_name in NETWORKS.items():
        G = load_baseline(name)
        print(f"\n=== {name} (N={G.number_of_nodes()}, E={G.number_of_edges()}) ===")

        with open(f"{META_DIR}{meta_name}_blocks.json") as f:
            block_map = json.load(f)

        original_nodes = list(G.nodes())
        g, _, idx_to_node = to_graph_tool(G)
        b_array = np.array([block_map[node] for node in original_nodes])

        for rep in range(N_REPLICATES):
            # seed_rng(0) means "seed from system entropy" in graph-tool,
            # which is not reproducible, so seeds start at 1
            gt.seed_rng(rep + 1)
            np.random.seed(rep + 1)
            null_nx = generate_null_edgelist(
                g, b_array, idx_to_node, original_nodes, n_iter=N_REWIRE_ITER
            )
            nx.write_edgelist(
                null_nx, f"{OUT_DIR}{name}_sbm_rep{rep}.csv", delimiter=",", data=False
            )
            print(f"  rep {rep}: E={null_nx.number_of_edges()}", flush=True)
