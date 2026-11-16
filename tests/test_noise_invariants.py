"""
Invariants of the perturbed networks (NoiseNetworks) and of reading them back
(utils.perturbed_io). These must hold for every noise type, whatever the
random numbers are: an error here would silently change every result
computed from the perturbed networks.
"""

import filecmp
import os

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import ISOLATED, LEVEL, N_REPEATS, edge_set, write_edgelist

from NoiseEffect.NoiseNetworks import ALL_NOISE_TYPES, generatePerturbedNetworks
from NoiseEffect.NoiseNetworks.main import ADDITIONS, REMOVALS, REWIRINGS
from NoiseEffect.utils.perturbed_io import (
    iter_perturbed_networks,
    load_node_index,
    parse_perturbed_path,
)


def networks(path, toy):
    return list(iter_perturbed_networks(path, toy["node_to_idx"]))


def degrees(edges, n_nodes):
    return np.bincount(edges.ravel(), minlength=n_nodes)


@pytest.mark.parametrize("noise_type", ALL_NOISE_TYPES)
def test_every_network_is_a_simple_graph_with_the_exact_edge_count(noise_type, toy, perturbed):
    n_modify = int(toy["n_edges"] * LEVEL)
    expected = toy["n_edges"] + {
        **dict.fromkeys(ADDITIONS, n_modify),
        **dict.fromkeys(REMOVALS, -n_modify),
        **dict.fromkeys(REWIRINGS, 0),
    }[noise_type]

    nets = networks(perturbed[noise_type], toy)
    assert [(replicate, repeat) for replicate, repeat, _ in nets] == [
        (0, repeat) for repeat in range(N_REPEATS)
    ]
    for _, _, edges in nets:
        assert len(edges) == expected
        assert (edges[:, 0] != edges[:, 1]).all(), "self-loop"
        assert len(edge_set(edges)) == len(edges), "duplicate edge"


@pytest.mark.parametrize("noise_type", ALL_NOISE_TYPES)
def test_only_the_intended_edges_change(noise_type, toy, perturbed):
    baseline = edge_set(toy["edges"])
    n_modify = int(toy["n_edges"] * LEVEL)
    for _, _, edges in networks(perturbed[noise_type], toy):
        new = edge_set(edges)
        if noise_type in ADDITIONS:
            assert baseline <= new
        elif noise_type in REMOVALS:
            assert new <= baseline
        else:
            # a swap can replace two original edges at once: one more at most
            assert n_modify <= len(baseline - new) <= n_modify + 1
            assert (
                degrees(edges, toy["n_nodes"]) == degrees(toy["edges"], toy["n_nodes"])
            ).all()


@pytest.mark.parametrize("noise_type", ALL_NOISE_TYPES)
def test_repeats_are_different_networks(noise_type, toy, perturbed):
    sets = [edge_set(edges) for _, _, edges in networks(perturbed[noise_type], toy)]
    assert sets[0] != sets[1] != sets[2] != sets[0]


def test_triadic_closure_only_closes_open_triangles(toy, perturbed):
    neighbours = {}
    for u, v in toy["edges"].tolist():
        neighbours.setdefault(u, set()).add(v)
        neighbours.setdefault(v, set()).add(u)
    baseline = edge_set(toy["edges"])
    for _, _, edges in networks(perturbed["triadic_closure_addition"], toy):
        for u, v in edge_set(edges) - baseline:
            assert neighbours[u] & neighbours[v], f"no common neighbour: {u}, {v}"


def test_hub_and_periphery_noise_target_the_right_nodes(toy, perturbed):
    """Mean baseline degree of the endpoints of the changed edges: hubs > random > periphery."""
    baseline = edge_set(toy["edges"])
    degree = degrees(toy["edges"], toy["n_nodes"])

    def mean_degree(noise_type):
        changed = []
        for _, _, edges in networks(perturbed[noise_type], toy):
            changed += list(edge_set(edges) ^ baseline)
        return degree[np.array(changed)].mean()

    for action in ("addition", "removal"):
        hub, random, periphery = (
            mean_degree(f"{target}_{action}") for target in ("hub", "random", "periphery")
        )
        assert hub > random > periphery, action


def test_isolated_nodes_stay_in_the_node_index(toy):
    assert toy["n_nodes"] == toy["graph"].number_of_nodes() + len(ISOLATED)
    assert set(ISOLATED) <= set(toy["node_to_idx"])
    # nodes without edges are not in the files, but the index keeps their place
    assert sorted(toy["node_to_idx"].values()) == list(range(toy["n_nodes"]))


def test_same_seed_gives_identical_files(toy, tmp_path):
    levels = {noise_type: [0.2] for noise_type in ALL_NOISE_TYPES}
    baselines = [(toy["edgelist"], toy["nodelist"])]
    for folder, seed in (("a", 7), ("b", 7), ("c", 8)):
        generatePerturbedNetworks(baselines, str(tmp_path / folder), levels, 2, seed=seed)
    for name in os.listdir(tmp_path / "a"):
        assert filecmp.cmp(tmp_path / "a" / name, tmp_path / "b" / name, shallow=False), name
        assert not filecmp.cmp(tmp_path / "a" / name, tmp_path / "c" / name, shallow=False), name


def test_a_file_can_be_regenerated_on_its_own(toy, tmp_path):
    """Every (type, level) has its own seed, so it does not depend on the other files."""
    baselines = [(toy["edgelist"], toy["nodelist"])]
    generatePerturbedNetworks(
        baselines, str(tmp_path / "all"), {t: [0.1, 0.2] for t in ALL_NOISE_TYPES}, 2
    )
    generatePerturbedNetworks(baselines, str(tmp_path / "one"), {"hub_removal": [0.2]}, 2)
    name = "hub_removal_0p2.parquet"
    assert filecmp.cmp(tmp_path / "all" / name, tmp_path / "one" / name, shallow=False)


def test_null_model_replicates(toy, tmp_path):
    """Several baselines (null model draws) end up as replicates of one file."""
    shuffled = toy["graph"].copy()
    nx.double_edge_swap(shuffled, nswap=200, max_tries=10_000, seed=3)
    second = write_edgelist(shuffled, tmp_path / "draw_1.csv")
    generatePerturbedNetworks(
        [(toy["edgelist"], toy["nodelist"]), (second, toy["nodelist"])],
        str(tmp_path / "toy" / "conf"),
        {"random_removal": [0.5]},
        num_repeats=2,
    )
    path = str(tmp_path / "toy" / "conf" / "random_removal_0p5.parquet")
    nets = networks(path, toy)
    assert [(replicate, repeat) for replicate, repeat, _ in nets] == [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ]
    originals = [edge_set(toy["edges"]), {
        tuple(sorted((toy["node_to_idx"][u], toy["node_to_idx"][v])))
        for u, v in shuffled.edges()
    }]
    for replicate, _, edges in nets:
        assert edge_set(edges) <= originals[replicate]


def test_existing_files_are_never_overwritten(toy, perturbed):
    folder = os.path.dirname(perturbed["random_removal"])
    with pytest.raises(FileExistsError):
        generatePerturbedNetworks(
            [(toy["edgelist"], toy["nodelist"])], folder, {"random_removal": [LEVEL]}, 1
        )


def test_invalid_requests_raise(toy, tmp_path):
    baselines = [(toy["edgelist"], toy["nodelist"])]
    with pytest.raises(ValueError, match="Unknown noise types"):
        generatePerturbedNetworks(baselines, str(tmp_path / "a"), {"hubs": [0.1]}, 1)
    with pytest.raises(ValueError, match="remove all edges"):
        generatePerturbedNetworks(baselines, str(tmp_path / "b"), {"random_removal": [1.0]}, 1)
    with pytest.raises(ValueError, match="modifies no edges"):
        generatePerturbedNetworks(baselines, str(tmp_path / "c"), {"random_removal": [1e-6]}, 1)


def test_unreachable_triadic_closure_levels_are_skipped(tmp_path):
    # a path of 5 nodes has 4 edges but only 3 pairs with a common neighbour
    edgelist = write_edgelist(nx.path_graph(5), tmp_path / "path.csv")
    skipped = generatePerturbedNetworks(
        [(edgelist, None)],
        str(tmp_path / "out"),
        {"triadic_closure_addition": [0.5, 1.0]},
        num_repeats=1,
    )
    assert skipped == [("triadic_closure_addition", 1.0)]
    assert os.listdir(tmp_path / "out") == ["triadic_closure_addition_0p5.parquet"]


def test_parse_perturbed_path():
    assert parse_perturbed_path("data/x/huri/er/hub_addition_0p1.parquet") == {
        "network": "huri",
        "model": "er",
        "noise_type": "hub_addition",
        "target": "hub",
        "action": "addition",
        "level": 0.1,
    }
    info = parse_perturbed_path("a/wiki/empirical/degree_preserving_rewiring_0p05.parquet")
    assert (info["target"], info["action"], info["level"]) == ("degree_preserving", "rewiring", 0.05)
    with pytest.raises(ValueError):
        parse_perturbed_path("a/wiki/empirical/results_0p1.parquet")


def test_reading_fails_on_nodes_outside_the_baseline(toy, tmp_path):
    path = str(tmp_path / "random_removal_0p1.parquet")
    table = pa.table(
        {
            "source": ["n000", "n001"],
            "target": ["n002", "not_a_node"],
            "replicate": pa.array([0, 0], pa.uint8()),
            "repeat": pa.array([0, 0], pa.uint8()),
        }
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="not in the baseline"):
        list(iter_perturbed_networks(path, toy["node_to_idx"]))


def test_a_network_spanning_row_groups_is_read_as_one(toy, tmp_path):
    path = str(tmp_path / "random_removal_0p1.parquet")
    table = pa.table(
        {
            "source": ["n000", "n001", "n002", "n003"],
            "target": ["n010", "n011", "n012", "n013"],
            "replicate": pa.array([0, 0, 0, 0], pa.uint8()),
            "repeat": pa.array([0, 0, 0, 1], pa.uint8()),
        }
    )
    pq.write_table(table, path, row_group_size=2)
    nets = list(iter_perturbed_networks(path, toy["node_to_idx"]))
    assert [(replicate, repeat, len(edges)) for replicate, repeat, edges in nets] == [
        (0, 0, 3), (0, 1, 1)
    ]


def test_node_index_is_sorted_and_matches_the_generator(toy):
    labels = sorted(list(toy["graph"].nodes()) + ISOLATED)
    assert load_node_index(toy["edgelist"], toy["nodelist"]) == {
        label: i for i, label in enumerate(labels)
    }
