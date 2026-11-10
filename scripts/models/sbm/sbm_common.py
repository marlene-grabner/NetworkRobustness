#! /usr/bin/env python3
"""Shared by the two SBM scripts in this folder (fitting and generating)."""

import graph_tool.all as gt
import networkx as nx

BASELINE_DIR = "./data/baseline_networks/"
META_DIR = BASELINE_DIR + "/null_models/metadata/"

# output name -> name of the block + metadata files
NETWORKS = {
    "power": "western_us_power_grid",
    "astro": "ca-AstroPh",
    "wiki": "wiki-Vote",
    "huri": "huri",
    "ppi": "chloe_ppi",
}


def load_baseline(name):
    """Load a baseline network as a simple graph."""
    G = nx.read_edgelist(f"{BASELINE_DIR}{name}.csv", delimiter=",")

    # SELF-LOOPS: astro.csv is the only baseline with self-loops (59). They are
    # removed here, because the SBM has to be a simple graph: the astro SBM
    # therefore has 196,972 edges, not the 197,031 of astro.csv.
    n_self_loops = nx.number_of_selfloops(G)
    if n_self_loops:
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        print(f"  removed {n_self_loops} self-loops from {name}")
    return G


def to_graph_tool(G):
    """Build the graph-tool graph, keeping a stable node<->index map."""
    original_nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(original_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    g = gt.Graph(directed=False)
    g.add_vertex(len(original_nodes))
    g.add_edge_list([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()])
    return g, node_to_idx, idx_to_node
