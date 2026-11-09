#!/usr/bin/env python3
"""
SBM ensemble replicates, WITHOUT refitting.

The expensive step in the production SBM script is the nested fit. The fitted
block assignment is already saved as <name>_blocks.json, so additional
realizations of the same ensemble only require re-running generate_sbm with
that fixed partition. This script does that and writes plain edgelists, which
ensemble_baseline.py then profiles in the normal venv via --extra-null-dir.

Run inside the graph-tool conda environment.

    python regenerate_sbm_ensemble.py --replicates 20

The block assignment is held FIXED across replicates by design: the SBM null
is defined as "this mesoscale partition, rewired", so the ensemble should vary
the rewiring, not the partition. Fit uncertainty is a separate question, and
if you want it, refit with different restart seeds instead.
"""

import argparse
import json
import os

import graph_tool.all as gt
import networkx as nx
import numpy as np


def sbm_replicate(G, block_map, seed):
    original_nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(original_nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    g = gt.Graph(directed=False)
    g.add_vertex(len(original_nodes))
    g.add_edge_list([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()])

    b = g.new_vertex_property("int")
    for n in original_nodes:
        b.a[node_to_idx[n]] = block_map[str(n)]

    state = gt.BlockState(g, b=b, deg_corr=True)
    e_matrix = state.get_matrix()
    degrees = g.degree_property_map("total").a

    gt.seed_rng(seed)
    np.random.seed(seed)
    null_gt = gt.generate_sbm(
        b.a,
        e_matrix,
        out_degs=degrees,
        micro_ers=False,
        micro_degs=False,
        directed=False,
    )

    null_nx = nx.Graph()
    null_nx.add_nodes_from(original_nodes)
    null_nx.add_edges_from(
        (idx_to_node[int(e.source())], idx_to_node[int(e.target())])
        for e in null_gt.edges()
    )
    return null_nx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline-dir", default="./data/baseline_networks/old_tsv_networks/"
    )
    ap.add_argument(
        "--meta-dir",
        default="./data/baseline_networks/old_tsv_networks/null_models/metadata/",
    )
    ap.add_argument("--out-dir", default="./data/ensemble_baseline/sbm_reps/")
    ap.add_argument("--replicates", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # (short key used downstream, production name, edgelist file)
    networks = [
        ("ppi", "chloe_ppi", "chloe_ppi_lcc_2026_02_23.tsv"),
        ("power", "western_us_power_grid", "western_us_power_grid.tsv"),
        ("astro", "ca-AstroPh", "ca-AstroPh_gcc.tsv"),
        ("wiki", "wiki-Vote", "wiki-Vote_gcc.tsv"),
    ]

    for key, prod_name, fname in networks:
        G = nx.read_edgelist(os.path.join(args.baseline_dir, fname), delimiter="\t")
        with open(os.path.join(args.meta_dir, f"{prod_name}_blocks.json")) as f:
            block_map = json.load(f)
        print(
            f"=== {key}: N={G.number_of_nodes()} M={G.number_of_edges()}, "
            f"{len(set(block_map.values()))} blocks ==="
        )

        for k in range(args.replicates):
            null = sbm_replicate(G, block_map, seed=k)
            out = os.path.join(args.out_dir, f"{key}_SBM_rep{k}.tsv")
            nx.write_edgelist(null, out, delimiter="\t", data=False)
            print(
                f"  rep {k + 1:2d}/{args.replicates}: "
                f"N={null.number_of_nodes()} M={null.number_of_edges()}"
            )
