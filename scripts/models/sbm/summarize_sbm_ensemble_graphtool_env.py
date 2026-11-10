#!/usr/bin/env python3
"""
Step 3 (checks): summarize the block fits and the SBM null model ensemble.

Writes two LaTeX tables (+ csvs) to outputs/:
  sbm_block_fits        from the *_meta.json files of fit_sbm_blocks_graphtool_env.py:
                        blocks, blocks/N, Delta L, Delta L per added block,
                        entropy SD across restarts
  sbm_ensemble_summary  per network, mean +- SD over the N_REPLICATES SBMs:
                        shared edges and Jaccard with the empirical network,
                        GCC fraction, components, and whether the degrees and
                        block-pair edge counts are exactly kept. Plus the
                        block modularity Q_B and the block type.

Block type: the edges above the degree-only expectation (d_r d_s / 2E between
blocks r and s) are summed per block pair. If at least ASSORTATIVE_MIN of
that excess lies inside blocks, the blocks are "Assortative", otherwise
"Role-like" (blocks are defined by which other blocks they link to).

# ====================================================================
# DISCLAIMER: requires the graph-tool package (use graphtool-env).
# Run from the repository root:
#   conda run -n graphtool-env python scripts/models/sbm/summarize_sbm_ensemble_graphtool_env.py
# ====================================================================
"""

import json
import os

import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd

from sbm_common import BASELINE_DIR, META_DIR, NETWORKS, load_baseline, to_graph_tool

ENSEMBLE_DIR = BASELINE_DIR + "null_models/sbm_ensemble/"
N_REPLICATES = 10
ASSORTATIVE_MIN = 0.2

CSV_DIR = "outputs/models/"
TEX_DIR = "outputs/latex_tables/models/sbm/"


def block_matrix(g, b):
    return gt.BlockState(g, b=b).get_matrix().tocsr()


def within_block_excess_share(e, degrees_per_block, n_edges):
    """Share of the above-expectation edges that lie inside blocks."""
    e = e.tocoo()
    inside, total = 0.0, 0.0
    for r, s, ers in zip(e.row, e.col, e.data):
        if r > s:
            continue
        # get_matrix counts within-block edges twice (both ends), between once per direction
        observed = ers / 2 if r == s else ers
        expected = degrees_per_block[r] * degrees_per_block[s] / (2 * n_edges)
        if r == s:
            expected /= 2
        excess = max(0.0, observed - expected)
        total += excess
        if r == s:
            inside += excess
    return inside / total


def load_null(name, rep, nodes):
    H = nx.read_edgelist(f"{ENSEMBLE_DIR}{name}_sbm_rep{rep}.csv", delimiter=",")
    H.add_nodes_from(nodes)
    return H


def summarize_network(name, meta_name):
    G = load_baseline(name)
    nodes = list(G.nodes())
    g, node_to_idx, _ = to_graph_tool(G)
    with open(f"{META_DIR}{meta_name}_blocks.json") as f:
        block_map = json.load(f)
    b = g.new_vertex_property("int", vals=np.array([block_map[n] for n in nodes]))

    e_orig = block_matrix(g, b)
    # undirected convention: every edge is counted at both ends, so the matrix sums to 2E
    assert e_orig.sum() == 2 * g.num_edges()
    deg = g.degree_property_map("total").a
    deg_per_block = np.bincount(b.a, weights=deg)
    q_b = gt.modularity(g, b)
    share_inside = within_block_excess_share(e_orig, deg_per_block, g.num_edges())

    emp_edges = {frozenset(e) for e in G.edges()}
    rows = []
    for rep in range(N_REPLICATES):
        H = load_null(name, rep, nodes)
        null_edges = {frozenset(e) for e in H.edges()}
        shared = len(emp_edges & null_edges)

        # same vertex indices as g, so the block labels line up
        h = gt.Graph(directed=False)
        h.add_vertex(len(nodes))
        h.add_edge_list([(node_to_idx[u], node_to_idx[v]) for u, v in H.edges()])
        b_h = h.new_vertex_property("int", vals=b.a)
        e_null = block_matrix(h, b_h)

        comps = list(nx.connected_components(H))
        rows.append(
            {
                "network": name,
                "rep": rep,
                "edges": H.number_of_edges(),
                "self_loops": nx.number_of_selfloops(H),
                "shared_pct": 100 * shared / len(emp_edges),
                "jaccard": shared / len(emp_edges | null_edges),
                "gcc_fraction": max(map(len, comps)) / H.number_of_nodes(),
                "components": len(comps),
                "isolated": sum(1 for _, d in H.degree() if d == 0),
                "degrees_changed": int(
                    sum(G.degree(n) != H.degree(n) for n in nodes)
                ),
                "block_pairs_changed": int((e_orig != e_null).nnz),
                "q_b_null": gt.modularity(h, b_h),
            }
        )
        print(f"  {name} rep {rep} done", flush=True)

    per_network = {
        "network": name,
        "nodes": len(nodes),
        "n_blocks": int(len(np.unique(b.a))),
        "q_b": q_b,
        "share_excess_inside": share_inside,
        "type": "Assortative" if share_inside >= ASSORTATIVE_MIN else "Role-like",
        # e_orig is symmetric with every between-block pair stored twice
        "block_pairs": int((e_orig.nnz + (e_orig.diagonal() != 0).sum()) // 2),
    }
    return per_network, rows


def pm(values, fmt):
    m, s = np.mean(values), np.std(values, ddof=1)
    return f"{fmt.format(m)} $\\pm$ {fmt.format(s)}"


def write_tex(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def block_fit_table():
    rows = []
    for name, meta_name in NETWORKS.items():
        with open(f"{META_DIR}{meta_name}_meta.json") as f:
            m = json.load(f)
        rows.append(
            {
                "network": name,
                "n_blocks": m["n_blocks"],
                "blocks_per_node_pct": 100 * m["n_blocks"] / m["orig_nodes"],
                "delta_L": m["delta_L"],
                "delta_L_per_block": m["delta_L_per_block"],
                "entropy_sd_restarts": m["entropy_std_across_restarts"],
                "best_entropy": m["best_entropy"],
                "n_levels": m["n_levels"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(CSV_DIR + "sbm_block_fits.csv", index=False)

    body = "\n".join(
        f"    {r.network} & {r.n_blocks:,} & {r.blocks_per_node_pct:.1f}\\% & "
        f"{r.delta_L:,.0f} & {r.delta_L_per_block:,.0f} & {r.entropy_sd_restarts:,.0f} \\\\"
        for r in df.itertuples()
    )
    write_tex(
        TEX_DIR + "sbm_block_fits.tex",
        "\\begin{table}[H]\n  \\centering\n"
        "  \\caption{Fitted block structure (nested degree-corrected SBM, best of 20 restarts "
        "with 1000 MCMC sweeps each, level 0). $B$: number of blocks. $\\Delta L$: description "
        "length saved by the fitted blocks compared to a single block, scored with the same flat "
        "degree-corrected model. $\\Delta L$ per block: $\\Delta L / \\max(B-1, 1)$. Entropy SD: "
        "standard deviation of the description length across the 20 restarts.}\n"
        "  \\label{tab:sbm_block_fits}\n"
        "  \\begin{tabular}{lrrrrr}\n    \\hline\n"
        "    Network & $B$ & $B/N$ & $\\Delta L$ (nats) & \\shortstack{$\\Delta L$ per block\\\\(nats)} "
        "& \\shortstack{Entropy SD\\\\(nats)} \\\\\n    \\hline\n"
        f"{body}\n    \\hline\n  \\end{{tabular}}\n\\end{{table}}\n",
    )


if __name__ == "__main__":
    os.makedirs(CSV_DIR, exist_ok=True)
    block_fit_table()

    networks, reps = [], []
    for name, meta_name in NETWORKS.items():
        per_network, rows = summarize_network(name, meta_name)
        networks.append(per_network)
        reps.extend(rows)
    net_df = pd.DataFrame(networks)
    rep_df = pd.DataFrame(reps)
    net_df.to_csv(CSV_DIR + "sbm_ensemble_networks.csv", index=False)
    rep_df.to_csv(CSV_DIR + "sbm_ensemble_per_rep.csv", index=False)
    print(net_df.to_string())
    print(rep_df.groupby("network").agg(["mean", "std", "min", "max"]).T.to_string())

    lines = []
    for r in net_df.itertuples():
        x = rep_df[rep_df.network == r.network]
        lines.append(
            f"    {r.network} & {r.n_blocks:,} & {r.q_b:.3f} & {r.type} & "
            f"{pm(x.shared_pct, '{:.1f}')}\\% & {pm(x.jaccard, '{:.3f}')} & "
            f"{pm(x.gcc_fraction, '{:.3f}')} & {pm(x.components, '{:.1f}')} & "
            f"{x.degrees_changed.max()} / {r.nodes:,} & "
            f"{x.block_pairs_changed.max()} / {r.block_pairs:,} \\\\"
        )
    write_tex(
        TEX_DIR + "sbm_ensemble_summary.tex",
        "\\begin{table}[H]\n  \\centering\n  \\footnotesize\n"
        f"  \\caption{{Microcanonical degree-corrected SBM null models, {N_REPLICATES} replicates "
        "per network (mean $\\pm$ SD over replicates). $B$: number of fitted blocks. $Q_B$: "
        "modularity under the fitted blocks (identical in the empirical network and every "
        "replicate, because the block-pair edge counts are fixed). Type: what the blocks capture, "
        "see text. Shared: percentage of empirical edges also present in the SBM. Jaccard: edge "
        "overlap between SBM and empirical network. GCC: fraction of all nodes in the giant "
        "component. Comp.: number of connected components. Degrees changed and block pairs "
        "changed: largest number, over all replicates, of nodes whose degree and of block pairs "
        "whose edge count differ from the empirical network.}\n"
        "  \\label{tab:sbm_microcanonical}\n  \\setlength{\\tabcolsep}{3pt}\n"
        "  \\resizebox{\\textwidth}{!}{%\n  \\begin{tabular}{lrrlrrrrrr}\n    \\hline\n"
        "    Network & $B$ & $Q_B$ & Type & Shared & Jaccard & GCC & Comp. & "
        "\\shortstack{Degrees\\\\changed} & \\shortstack{Block pairs\\\\changed} \\\\\n    \\hline\n"
        + "\n".join(lines)
        + "\n    \\hline\n  \\end{tabular}}\n\\end{table}\n",
    )
