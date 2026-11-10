#!/usr/bin/env python3
"""
Step 1 of 2: fit the block structure of the five main networks.

For each empirical network: fit a nested degree-corrected SBM, with N_RESTARTS
independent attempts, each refined by MCMC sweeps; keep the lowest-entropy fit;
check it against the B=1 baseline; save the block assignment and a metadata
record. The null models are made from these blocks by
generate_sbm_null_models_graphtool_env.py.

Networks that already have a block file are skipped. To refit one, move its
{name}_blocks.json and {name}_meta.json out of the metadata folder first.

# ====================================================================
# DISCLAIMER: requires the graph-tool package (use graphtool-env).
# Run from the repository root:
#   conda run -n graphtool-env python scripts/models/sbm/fit_sbm_blocks_graphtool_env.py
# ====================================================================
"""

import json
import os

import graph_tool.all as gt
import numpy as np

from sbm_common import META_DIR, NETWORKS, load_baseline, to_graph_tool

N_RESTARTS = 20
N_MCMC_SWEEPS = 1000


def fit_best_sbm(
    g,
    n_restarts: int = 15,
    n_mcmc_sweeps: int = 200,
    deg_corr: bool = True,
    verbose: bool = True,
):
    """
    Fit a nested degree-corrected SBM with multiple restarts + MCMC
    refinement. Returns the lowest-entropy fit and the entropy of every restart.
    """
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

    return best_state, base_level, entropies


def significance_vs_degree(g, base_level, deg_corr: bool = True):
    """
    Does the fitted partition earn its keep vs. assuming NO block structure?
    Compares description length of the fit against a single-block (B=1),
    degree-corrected baseline -- the SBM equivalent of the configuration model.
    Both partitions are scored with the same flat model.

    Positive, large delta_L  -> blocks capture real structure beyond degree.
    Near-zero / negative      -> partition is not meaningfully better than
                                 'degree sequence alone' (degenerate signal).
    """
    B_fit = base_level.get_nonempty_B()

    b_fit = g.new_vertex_property("int", vals=np.array(base_level.b.a))
    L_fit = gt.BlockState(g, b=b_fit, deg_corr=deg_corr).entropy()

    b_trivial = g.new_vertex_property("int", vals=np.zeros(g.num_vertices(), dtype=int))
    L_b1 = gt.BlockState(g, b=b_trivial, deg_corr=deg_corr).entropy()

    delta_L = L_b1 - L_fit
    return {
        "L_fit": float(L_fit),
        "L_b1": float(L_b1),
        "delta_L": float(delta_L),
        "delta_L_per_block": float(delta_L / max(B_fit - 1, 1)),
        "n_blocks": int(B_fit),
    }


if __name__ == "__main__":
    for name, meta_name in NETWORKS.items():
        blocks_path = f"{META_DIR}{meta_name}_blocks.json"
        if os.path.exists(blocks_path):
            print(f"{name}: {blocks_path} already exists, skipping the fit.")
            continue

        G = load_baseline(name)
        print(f"\n=== {name} (N={G.number_of_nodes()}, E={G.number_of_edges()}) ===")
        g, node_to_idx, _ = to_graph_tool(G)

        best_state, base_level, entropies = fit_best_sbm(
            g, n_restarts=N_RESTARTS, n_mcmc_sweeps=N_MCMC_SWEEPS
        )

        sig = significance_vs_degree(g, base_level)
        print(
            f"  significance: delta_L={sig['delta_L']:.1f} "
            f"(per block {sig['delta_L_per_block']:.2f}), blocks={sig['n_blocks']}"
        )
        if sig["delta_L"] <= 0:
            print("  !! WARNING: partition NOT better than degree-only baseline.")

        # BLOCK ASSIGNMENT -- lets you regenerate null models without refitting
        block_map = {
            str(node): int(base_level.b.a[idx]) for node, idx in node_to_idx.items()
        }
        with open(blocks_path, "w") as f:
            json.dump(block_map, f)

        # METADATA -- everything the methods section / a reviewer needs
        meta = {
            "network": meta_name,
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
            **sig,
        }
        with open(f"{META_DIR}{meta_name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  saved blocks + metadata for {name}")
