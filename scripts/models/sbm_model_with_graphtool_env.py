#!/usr/bin/env python3
"""
Production SBM null-model generation.

For each empirical network:
  1. Fit a nested degree-corrected SBM, with N_RESTARTS independent attempts,
     each refined by MCMC sweeps; keep the lowest-entropy (best) fit.
  2. Run a significance check (fitted vs. B=1 baseline) to confirm the
     partition captures real structure beyond degree sequence.
  3. Generate the degree-preserving SBM null model from the best fit.
  4. Save: null-model edgelist (for the downstream pipeline) + a metadata
     record (for reproducibility / the methods table) + the block assignment.

NOTE: written against standard graph-tool API conventions; not executed here
(no graph-tool in this environment). Smoke-test on ONE network with
N_RESTARTS=2, N_MCMC_SWEEPS=10 before committing to the full run.

# ====================================================================
# DISCLAIMER: requires the graph-tool package (use graphtool-env).
# ====================================================================
"""

import graph_tool.all as gt
import networkx as nx
import numpy as np
import json
import os


def fit_best_sbm(
    G: nx.Graph,
    n_restarts: int = 15,
    n_mcmc_sweeps: int = 200,
    deg_corr: bool = True,
    verbose: bool = True,
):
    """
    Fit a nested degree-corrected SBM with multiple restarts + MCMC
    refinement. Returns the lowest-entropy fit and everything needed to
    generate the null and to audit the fit later.
    """
    # --- Build the graph-tool graph, keeping a stable node<->index map ---
    original_nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(original_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    g = gt.Graph(directed=False)
    g.add_vertex(len(original_nodes))
    g.add_edge_list([(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()])

    best_state = None
    best_entropy = np.inf
    entropies = []

    for r in range(n_restarts):
        # Each restart begins from an independent random initialization,
        # so different restarts explore different regions of the landscape.
        state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=deg_corr))

        # Zero-temperature (beta=inf) MCMC refinement: only accept moves that
        # LOWER description length. This walks the fit downhill out of the
        # poor local optimum that a single minimize_*_dl call often lands in.
        for _ in range(n_mcmc_sweeps):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=1)

        ent = state.entropy()
        entropies.append(ent)
        if verbose:
            b0 = state.get_levels()[0].get_nonempty_B()
            print(
                f"    restart {r + 1:2d}/{n_restarts}: entropy={ent:.2f}, L0 blocks={b0}"
            )

        if ent < best_entropy:
            best_entropy = ent
            best_state = state

    base_level = best_state.get_levels()[0]  # flat level-0 partition
    if verbose:
        print(
            f"  BEST: entropy={best_entropy:.2f}, "
            f"L0 blocks={base_level.get_nonempty_B()}, "
            f"levels={len(best_state.get_levels())}, "
            f"entropy spread across restarts={np.std(entropies):.2f}"
        )

    return best_state, base_level, g, node_to_idx, idx_to_node, entropies


def significance_vs_degree(g, base_level, deg_corr: bool = True):
    """
    Does the fitted partition earn its keep vs. assuming NO block structure?
    Compares description length of the fit against a single-block (B=1),
    degree-corrected baseline -- the SBM equivalent of the configuration model.

    Positive, large delta_L  -> blocks capture real structure beyond degree.
    Near-zero / negative      -> partition is not meaningfully better than
                                 'degree sequence alone' (degenerate signal).
    """
    L_fit = base_level.entropy()
    B_fit = base_level.get_nonempty_B()

    b_trivial = np.zeros(g.num_vertices(), dtype=int)
    L_b1 = gt.BlockState(g, b=b_trivial, deg_corr=deg_corr).entropy()

    delta_L = L_b1 - L_fit
    return {
        "L_fit": float(L_fit),
        "L_b1": float(L_b1),
        "delta_L": float(delta_L),
        "delta_L_per_block": float(delta_L / max(B_fit - 1, 1)),
        "n_blocks": int(B_fit),
    }


def generate_null_edgelist(base_level, g, node_to_idx, idx_to_node, original_nodes):
    """Generate the degree-preserving SBM null model as a NetworkX graph."""
    b_array = base_level.b.a  # per-node block label
    e_matrix = base_level.get_matrix()  # block-to-block edge count matrix
    degrees = g.degree_property_map("total").a  # undirected -> "total"

    null_gt = gt.generate_sbm(
        b_array,
        e_matrix,
        out_degs=degrees,
        micro_ers=False,  # match expected block-mixing counts, not exact
        micro_degs=False,  # match expected degrees, not exact
        directed=False,
    )

    null_nx = nx.Graph()
    null_nx.add_nodes_from(original_nodes)  # preserve isolated/original nodes
    null_nx.add_edges_from(
        (idx_to_node[int(e.source())], idx_to_node[int(e.target())])
        for e in null_gt.edges()
    )
    return null_nx


# ====================================================================
# Run
# ====================================================================

if __name__ == "__main__":
    OUT_DIR = "./data/baseline_networks/null_models/"
    META_DIR = "./data/baseline_networks/null_models/metadata/"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    N_RESTARTS = 20
    N_MCMC_SWEEPS = 1000

    G_ppi = nx.read_edgelist(
        "./data/baseline_networks/chloe_ppi_lcc_2026_02_23.tsv", delimiter="\t"
    )
    G_power = nx.read_edgelist(
        "./data/baseline_networks/western_us_power_grid.tsv", delimiter="\t"
    )
    G_collab = nx.read_edgelist(
        "./data/baseline_networks/ca-AstroPh_gcc.tsv", delimiter="\t"
    )
    G_wiki = nx.read_edgelist(
        "./data/baseline_networks/wiki-Vote_gcc.tsv", delimiter="\t"
    )

    networks = {
        "chloe_ppi": G_ppi,
        "western_us_power_grid": G_power,
        "ca-AstroPh": G_collab,
        "wiki-Vote": G_wiki,
    }

    for name, G in networks.items():
        print(f"\n=== {name} (N={G.number_of_nodes()}, E={G.number_of_edges()}) ===")

        best_state, base_level, g, node_to_idx, idx_to_node, entropies = fit_best_sbm(
            G, n_restarts=N_RESTARTS, n_mcmc_sweeps=N_MCMC_SWEEPS
        )

        sig = significance_vs_degree(g, base_level)
        print(
            f"  significance: delta_L={sig['delta_L']:.1f} "
            f"(per block {sig['delta_L_per_block']:.2f}), blocks={sig['n_blocks']}"
        )
        if sig["delta_L"] <= 0:
            print("  !! WARNING: partition NOT better than degree-only baseline.")

        original_nodes = list(G.nodes())
        null_nx = generate_null_edgelist(
            base_level, g, node_to_idx, idx_to_node, original_nodes
        )
        print(
            f"  null model: N={null_nx.number_of_nodes()}, E={null_nx.number_of_edges()}"
        )

        # (1) EDGELIST -- what the downstream pipeline consumes
        nx.write_edgelist(
            null_nx, f"{OUT_DIR}{name}_sbm.tsv", delimiter="\t", data=False
        )

        # (2) BLOCK ASSIGNMENT -- lets you regenerate more null replicates or
        #     re-run diagnostics without refitting (fitting is the expensive part)
        block_map = {
            str(node): int(base_level.b.a[node_to_idx[node]]) for node in original_nodes
        }
        with open(f"{META_DIR}{name}_blocks.json", "w") as f:
            json.dump(block_map, f)

        # (3) METADATA -- everything the methods section / a reviewer needs
        meta = {
            "network": name,
            "model": "nested_degree_corrected_sbm",
            "n_restarts": N_RESTARTS,
            "n_mcmc_sweeps": N_MCMC_SWEEPS,
            "best_entropy": float(best_state.entropy()),
            "entropy_mean_across_restarts": float(np.mean(entropies)),
            "entropy_std_across_restarts": float(np.std(entropies)),
            "n_levels": len(best_state.get_levels()),
            "blocks_per_level": [
                int(l.get_nonempty_B()) for l in best_state.get_levels()
            ],
            "orig_nodes": G.number_of_nodes(),
            "orig_edges": G.number_of_edges(),
            "null_nodes": null_nx.number_of_nodes(),
            "null_edges": null_nx.number_of_edges(),
            **sig,
        }
        with open(f"{META_DIR}{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  saved edgelist + blocks + metadata for {name}")
