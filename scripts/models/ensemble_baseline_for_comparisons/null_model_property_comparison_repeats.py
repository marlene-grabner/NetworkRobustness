#!/usr/bin/env python3
"""
Baseline descriptors over null-model ENSEMBLES rather than single realizations.

For each empirical network this generates N_REPLICATES independent draws of
the ER and configuration ensembles, profiles each with get_network_profile,
and reports per-descriptor mean, standard deviation, and the z-score of the
empirical value against the ensemble.

If the single null realizations already used in the pipeline are supplied
(--single-null-dir), the script additionally reports where each of them falls
within its own ensemble, as a percentile. That is the check that certifies
the specific graphs used downstream are typical draws.

SBM ensembles are produced separately by regenerate_sbm_ensemble.py, which
needs graph-tool. Point --extra-null-dir at its output to fold those
realizations into the same summary.

Runtime note
------------
get_network_profile computes EXACT average path length, which is all-pairs
BFS: O(N*M). For AstroPh (N=17903, M=197031) that dominates everything else
and makes a 20-replicate sweep expensive. Use --path-length sampled (or none)
for the ensemble sweep and keep the exact value for the empirical networks;
the flag records which was used in the output so the two are never silently
mixed.
"""

import argparse
import json
import os
import random
import sys
import time

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd

from NoiseEffect import TopologicalProperties


# ----------------------------------------------------------------------
# Null model generators, seeded
# ----------------------------------------------------------------------


def erdos_renyi_model_from_graph(G: nx.Graph, seed: int) -> nx.Graph:
    """G(n, m) with the empirical node and edge counts.

    NOTE: the production version of this function hard-codes seed=0, so every
    call returns the same graph. The seed argument here is what makes an
    ensemble an ensemble.
    """
    n, m = G.number_of_nodes(), G.number_of_edges()
    mapping = dict(enumerate(G.nodes()))
    return nx.relabel_nodes(nx.gnm_random_graph(n, m, seed=seed), mapping)


def configuration_model_from_graph(G: nx.Graph, seed: int) -> nx.Graph:
    """Viger-Latapy: uniform over simple connected graphs with this degree
    sequence. igraph draws from its own RNG, so it is seeded explicitly."""
    original_nodes = list(G.nodes())
    degree_sequence = [G.degree(n) for n in original_nodes]
    ig.set_random_number_generator(random.Random(seed))
    g_ig = ig.Graph.Degree_Sequence(degree_sequence, method="vl")
    if sorted(g_ig.degree()) != sorted(degree_sequence):
        raise RuntimeError("VL sampler did not preserve the degree sequence")
    return nx.relabel_nodes(g_ig.to_networkx(), dict(enumerate(original_nodes)))


GENERATORS = {
    "ER": erdos_renyi_model_from_graph,
    "Conf": configuration_model_from_graph,
}

NUMERIC_DESCRIPTORS = [
    "Nodes_Full",
    "Edges_Full",
    "N_Components",
    "N_Singletons",
    "GCC_Fraction",
    "Nodes",
    "Edges",
    "Average_Degree",
    "Density",
    "Degree_Skew",
    "Degree_CV",
    "Assortativity",
    "Transitivity",
    "Clustering_Coefficient",
    "Modularity_Louvain",
    "Algebraic_Connectivity",
    "Spectral_Gap",
    "Avg_Path_Length",
]


def profile(G, label, path_length_mode, verbose=False):
    """get_network_profile, with the option to skip or sample path length."""
    if path_length_mode == "exact":
        return TopologicalProperties.get_network_profile(
            G, metrics_on_gcc=True, verbose=verbose
        )

    # Temporarily neutralise the exact APSP call, then fill the value in.
    prof = _profile_without_path_length(G, verbose=verbose)
    if path_length_mode == "none":
        prof["Avg_Path_Length"] = np.nan
    else:
        prof["Avg_Path_Length"] = _sampled_path_length(G, int(path_length_mode))
    return prof


def _profile_without_path_length(G, verbose=False):
    """Monkeypatch igraph's exact APSP for the duration of one profile call."""
    original = ig.Graph.average_path_length
    ig.Graph.average_path_length = lambda self, *a, **k: np.nan
    try:
        return TopologicalProperties.get_network_profile(
            G, metrics_on_gcc=True, verbose=verbose
        )
    finally:
        ig.Graph.average_path_length = original


def _sampled_path_length(G, n_sources, seed=0):
    """Mean shortest path from n_sources random sources in the GCC."""
    comp = max(nx.connected_components(G), key=len)
    H = G.subgraph(comp)
    rng = random.Random(seed)
    sources = rng.sample(list(H.nodes()), min(n_sources, H.number_of_nodes()))
    total = count = 0
    for s in sources:
        d = nx.single_source_shortest_path_length(H, s)
        total += sum(d.values())
        count += len(d) - 1
    return total / count if count else np.nan


# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline-dir", default="./data/baseline_networks/old_tsv_networks"
    )
    ap.add_argument("--out-dir", default="./data/ensemble_baseline/")
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument(
        "--path-length",
        default="500",
        help="'exact', 'none', or an integer number of BFS sources",
    )
    ap.add_argument(
        "--single-null-dir",
        default="./data/baseline_networks/old_tsv_networks/null_models/",
        help="directory of the single null edgelists already used "
        "downstream, named <key>_<model>.tsv",
    )
    ap.add_argument(
        "--extra-null-dir",
        default="./data/ensemble_baseline/sbm_reps/",
        help="directory of pre-generated ensembles (e.g. SBM), "
        "named <key>_<model>_rep<k>.tsv",
    )
    ap.add_argument("--models", nargs="+", default=["ER", "Conf"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    networks = {
        "ppi": "chloe_ppi_lcc_2026_02_23.tsv",
        "power": "western_us_power_grid.tsv",
        "astro": "ca-AstroPh_gcc.tsv",
        "wiki": "wiki-Vote_gcc.tsv",
    }

    rows = []
    for key, fname in networks.items():
        path = os.path.join(args.baseline_dir, fname)
        if not os.path.exists(path):
            print(f"!! missing {path}, skipping {key}")
            continue
        G = nx.read_edgelist(path, delimiter="\t")
        print(f"\n=== {key}: N={G.number_of_nodes()} M={G.number_of_edges()} ===")

        t0 = time.time()
        prof = profile(G, key, args.path_length)
        prof.update(network=key, model="empirical", replicate=-1)
        rows.append(prof)
        print(f"  empirical profiled in {time.time() - t0:.1f}s")

        for model in args.models:
            gen = GENERATORS[model]
            for k in range(args.replicates):
                t0 = time.time()
                Gn = gen(G, seed=k)
                prof = profile(Gn, f"{key}_{model}_{k}", args.path_length)
                prof.update(network=key, model=model, replicate=k)
                rows.append(prof)
                print(
                    f"  {model} rep {k + 1:2d}/{args.replicates} "
                    f"({time.time() - t0:.1f}s)"
                )

        if args.extra_null_dir:
            for f in sorted(os.listdir(args.extra_null_dir)):
                if not f.startswith(f"{key}_") or "_rep" not in f:
                    continue
                model = f.split("_")[1]
                k = int(f.rsplit("_rep", 1)[1].split(".")[0])
                Gn = nx.read_edgelist(
                    os.path.join(args.extra_null_dir, f), delimiter="\t"
                )
                prof = profile(Gn, f, args.path_length)
                prof.update(network=key, model=model, replicate=k)
                rows.append(prof)
                print(f"  {model} rep {k} (from disk)")

        if args.single_null_dir:
            for model in args.models + ["SBM"]:
                f = os.path.join(args.single_null_dir, f"{key}_{model}.tsv")
                if not os.path.exists(f):
                    continue
                Gn = nx.read_edgelist(f, delimiter="\t")
                prof = profile(Gn, f, args.path_length)
                prof.update(network=key, model=model, replicate=-2)  # the one used
                rows.append(prof)
                print(f"  {model} (single realization used downstream)")

    raw = pd.DataFrame(rows)
    raw["path_length_mode"] = args.path_length
    raw.to_csv(os.path.join(args.out_dir, "ensemble_profiles_raw.csv"), index=False)

    # ------------------------------------------------------------------
    # Summary: ensemble mean/std, z of empirical, percentile of the single
    # realization already used downstream.
    # ------------------------------------------------------------------
    summary = []
    for (net, model), grp in raw[raw.replicate >= 0].groupby(["network", "model"]):
        emp = raw[(raw.network == net) & (raw.model == "empirical")]
        single = raw[
            (raw.network == net) & (raw.model == model) & (raw.replicate == -2)
        ]
        for d in NUMERIC_DESCRIPTORS:
            if d not in grp:
                continue
            vals = pd.to_numeric(grp[d], errors="coerce").dropna().values
            if vals.size == 0:
                continue
            mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            e = float(emp[d].iloc[0]) if len(emp) else np.nan
            rec = dict(
                network=net,
                model=model,
                descriptor=d,
                n_replicates=vals.size,
                empirical=e,
                null_mean=mu,
                null_std=sd,
                null_cv=sd / abs(mu) if mu else np.nan,
                ratio_mean=mu / e if e else np.nan,
                z_empirical=(e - mu) / sd if sd > 0 else np.nan,
            )
            if len(single):
                s = float(single[d].iloc[0])
                rec["single_used"] = s
                rec["single_percentile"] = float((vals < s).mean() * 100)
                rec["single_z"] = (s - mu) / sd if sd > 0 else np.nan
            summary.append(rec)

    summ = pd.DataFrame(summary)
    summ.to_csv(os.path.join(args.out_dir, "ensemble_summary.csv"), index=False)

    # Flag anything where the single realization is atypical, or where the
    # ensemble is too disperse for a single draw to have been safe.
    if "single_z" in summ:
        odd = summ[summ.single_z.abs() > 2].dropna(subset=["single_z"])
        if len(odd):
            print("\n!! single realization more than 2 SD from ensemble mean:")
            print(
                odd[
                    [
                        "network",
                        "model",
                        "descriptor",
                        "single_used",
                        "null_mean",
                        "null_std",
                        "single_z",
                    ]
                ].to_string(index=False)
            )
        else:
            print("\nAll single realizations lie within 2 SD of their ensemble mean.")

    disperse = summ[(summ.null_cv > 0.10) & summ.null_cv.notna()]
    if len(disperse):
        print(
            "\nDescriptors with ensemble CV > 10% (a single draw is not "
            "representative here):"
        )
        print(
            disperse[
                ["network", "model", "descriptor", "null_mean", "null_std", "null_cv"]
            ].to_string(index=False)
        )

    print(f"\nwrote {args.out_dir}ensemble_profiles_raw.csv and ensemble_summary.csv")


if __name__ == "__main__":
    main()
