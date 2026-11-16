"""
Smoke test: the whole pipeline once on the toy network, the way the scripts
in scripts/ call it (generate -> global properties -> communities -> seed
expansion). It checks that everything runs and returns tables of the expected
shape with sane values; the numbers themselves are checked in
test_metric_references.py.
"""

import numpy as np
import pandas as pd
from conftest import N_REPEATS

from NoiseEffect import CommunityDetection as cd
from NoiseEffect import GlobalProperties as gp
from NoiseEffect.NoiseNetworks import ALL_NOISE_TYPES
from NoiseEffect.SeedExpansion import Expansion as se
from NoiseEffect.SeedExpansion import SyntheticSeeds as synth

ID_COLUMNS = ["network", "model", "noise_type", "target", "action", "level"]


def test_all_noise_types_are_generated(perturbed):
    assert set(perturbed) == set(ALL_NOISE_TYPES)
    assert len(perturbed) == 9


def test_global_properties(toy, perturbed):
    baseline = gp.global_properties_of_baseline(toy["edgelist"], toy["node_to_idx"])
    assert baseline["degree_rank_corr"] == 1.0
    assert baseline["pagerank_top10_overlap"] == 1.0
    assert baseline["num_singletons"] == 2

    for path in perturbed.values():
        df = gp.global_properties_of_file(path, toy["node_to_idx"], [toy["edgelist"]])
        assert len(df) == N_REPEATS
        assert list(df.columns[:8]) == ID_COLUMNS + ["replicate", "repeat"]
        assert (df.network == "toy").all() and (df.model == "empirical").all()
        assert sorted(df.repeat) == list(range(N_REPEATS))
        metric_columns = df.columns[8:]
        assert set(baseline) == set(metric_columns)
        assert not df[metric_columns].isna().any().any()
        assert df.gcc.between(0, 1).all() and df.efficiency.between(0, 1).all()


def test_community_detection(toy, perturbed):
    baselines = cd.baseline_partitions([toy["edgelist"]], toy["node_to_idx"])
    assert set(baselines) == {f"{a}_0" for a in cd.DEFAULT_ALGORITHMS}
    for labels in baselines.values():
        assert labels.shape == (len(cd.SEEDS), toy["n_nodes"])

    baseline_table = cd.community_properties_of_baselines(baselines)
    assert len(baseline_table) == len(cd.DEFAULT_ALGORITHMS)
    assert "ari_baseline_all" not in baseline_table

    df = cd.community_properties_of_file(
        perturbed["random_removal"], toy["node_to_idx"], baselines
    )
    assert len(df) == N_REPEATS * len(cd.DEFAULT_ALGORITHMS)
    assert set(df.algorithm) == set(cd.DEFAULT_ALGORITHMS)
    # 10% random removal leaves four dense communities clearly recoverable
    assert (df.ari_baseline_connected > 0.8).all()
    assert (df.n_valid_baseline_pairs == len(cd.SEEDS) ** 2).all()


def test_seed_expansion(toy, perturbed):
    adj = se.edges_to_adjacency(toy["edges"], toy["n_nodes"])
    records = synth.bfs_seed_sets(
        adj, [0.5], n_sets=3, module_size=20, seed_fraction=0.4,
        max_overlap=0.5, rng=np.random.default_rng(0),
    )
    assert len(records) == 3
    idx_to_node = {i: n for n, i in toy["node_to_idx"].items()}
    seed_table = pd.DataFrame(
        {
            "network": "toy",
            "model": "empirical",
            "replicate": 0,
            "set_id": set_id,
            "seed_type": "bfs",
            "seed_nodes": ";".join(idx_to_node[i] for i in record["seeds"]),
            "hidden_nodes": ";".join(idx_to_node[i] for i in record["hidden"]),
        }
        for set_id, record in enumerate(records)
    )
    seed_sets = se.seed_sets_of_unit(seed_table, "toy", "empirical", toy["node_to_idx"])

    k_list = [5, 10]
    baseline, tops = se.baseline_seed_expansion(
        [toy["edgelist"]], toy["node_to_idx"], seed_sets, k_list=k_list
    )
    n_rows = len(records) * len(se.DEFAULT_ALGORITHMS) * len(k_list)
    assert len(baseline) == n_rows
    assert len(tops) == len(records) * len(se.DEFAULT_ALGORITHMS)
    assert "jaccard" not in baseline  # no stability columns for the baseline
    # a BFS module is local, so every algorithm finds some of the hidden nodes
    assert (baseline.groupby("algorithm").recovery_recall.mean() > 0).all()

    df = se.seed_expansion_of_file(
        perturbed["random_removal"], toy["node_to_idx"], seed_sets, tops, k_list=k_list
    )
    assert len(df) == n_rows * N_REPEATS
    for column in ("jaccard", "precision", "recall", "f1", "auroc", "auprc",
                   "recovery_recall", "recovery_precision", "recovery_auroc"):
        assert df[column].between(0, 1).all(), column
