"""
Community detection on one network, with the same settings for the
unperturbed and the perturbed networks.
"""

import igraph as ig
import numpy as np

from .detection_algorithms import (
    infomapAlgorithmPartioning,
    labelPropagationPartitioning,
    leidenAlgorithmPartioning,
    louvainPartioning,
)

# algorithm -> function(graph, seeds) returning {seed: list of node sets}
ALGORITHMS = {
    # modularity; 2 iterations are the leidenalg default
    "leiden": lambda g, seeds: leidenAlgorithmPartioning(g, seeds, n_iterations=2),
    # map equation, two-level, 1 trial per seed
    "infomap": lambda g, seeds: infomapAlgorithmPartioning(g, seeds, n_iterations=1),
    "label_propagation": labelPropagationPartitioning,
    "louvain": louvainPartioning,  # modularity like leiden; not run by default
}
DEFAULT_ALGORITHMS = ("leiden", "infomap", "label_propagation")
SEEDS = tuple(range(1, 11))


def detect_partitions(
    edges: np.ndarray, n_nodes: int, algorithm: str, seeds=SEEDS
) -> np.ndarray:
    """
    Runs the algorithm once per seed. Returns the labels as an
    (n_seeds, n_nodes) array; nodes without edges get the label -1, whatever
    the algorithm does with them.

    :param edges: (m, 2) integer array, node indices in 0..n_nodes-1
    """
    g = ig.Graph(n=n_nodes, edges=edges.tolist(), directed=False).simplify()
    partitions = ALGORITHMS[algorithm](g, list(seeds))
    labels = np.full((len(seeds), n_nodes), -1, dtype=np.int32)
    for i, seed in enumerate(seeds):
        for community_id, community in enumerate(partitions[seed]):
            labels[i, list(community)] = community_id
    labels[:, np.array(g.degree()) == 0] = -1
    return labels
