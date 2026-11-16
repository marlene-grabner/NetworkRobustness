"""
Shared toy data for all tests: a small network with four planted communities
and two nodes without edges, and its perturbed networks (every noise type, one
level, three repeats). Everything is generated once per test run in a
temporary folder; no project data is needed.
"""

import os

import networkx as nx
import numpy as np
import pytest

from NoiseEffect.NoiseNetworks import ALL_NOISE_TYPES, generatePerturbedNetworks
from NoiseEffect.utils.perturbed_io import load_node_index, read_baseline_edges

N_COMMUNITIES, COMMUNITY_SIZE = 4, 30
ISOLATED = ["z_isolated_1", "z_isolated_2"]
LEVEL = 0.1
N_REPEATS = 3


def write_edgelist(g: nx.Graph, path) -> str:
    nx.write_edgelist(g, path, delimiter=",", data=False)
    return str(path)


@pytest.fixture(scope="session")
def toy_graph() -> nx.Graph:
    """Connected, 120 nodes with string labels n000..n119; community = index // 30."""
    g = nx.planted_partition_graph(N_COMMUNITIES, COMMUNITY_SIZE, 0.4, 0.02, seed=1)
    assert nx.is_connected(g)
    return nx.relabel_nodes(g, {n: f"n{n:03d}" for n in g})


@pytest.fixture(scope="session")
def toy(tmp_path_factory, toy_graph) -> dict:
    """The toy network on disk, as the pipeline reads it."""
    folder = tmp_path_factory.mktemp("toy")
    edgelist = write_edgelist(toy_graph, folder / "toy.csv")
    nodelist = folder / "toy_isolated_nodes.csv"
    nodelist.write_text(",".join(ISOLATED))
    node_to_idx = load_node_index(edgelist, str(nodelist))
    return {
        "folder": folder,
        "graph": toy_graph,
        "edgelist": edgelist,
        "nodelist": str(nodelist),
        "node_to_idx": node_to_idx,
        "n_nodes": len(node_to_idx),
        "edges": read_baseline_edges(edgelist, node_to_idx),
        "n_edges": toy_graph.number_of_edges(),
        "truth": np.arange(N_COMMUNITIES * COMMUNITY_SIZE) // COMMUNITY_SIZE,
    }


@pytest.fixture(scope="session")
def perturbed(toy) -> dict:
    """noise type -> perturbed file of the toy network (level 0.1, 3 repeats)."""
    folder = os.path.join(toy["folder"], "perturbed", "toy", "empirical")
    skipped = generatePerturbedNetworks(
        [(toy["edgelist"], toy["nodelist"])],
        folder,
        {noise_type: [LEVEL] for noise_type in ALL_NOISE_TYPES},
        num_repeats=N_REPEATS,
        seed=0,
    )
    assert skipped == []
    return {t: os.path.join(folder, f"{t}_0p1.parquet") for t in ALL_NOISE_TYPES}


def edge_set(edges: np.ndarray) -> set:
    """Undirected edges of an (m, 2) array as a set of sorted tuples."""
    return {(min(u, v), max(u, v)) for u, v in edges.tolist()}
