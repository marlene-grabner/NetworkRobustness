import numpy as np
import networkx as nx
from typing import Tuple, Dict
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import igraph as ig
import leidenalg as la


def extract_gcc(A: coo_matrix, node_to_idx: dict) -> Tuple[coo_matrix, dict, float]:
    """
    Identifies the Giant Connected Component (GCC) and slices the matrix.
    Returns the new GCC matrix, the updated node mapping, and the GCC Fraction
    (the percentage of nodes that survived).
    """
    n_original_nodes = A.shape[0]
    n_components, labels = connected_components(csgraph=A, directed=False)

    if n_components == 1:
        return A, node_to_idx, 1.0  # Network is perfectly connected

    unique_labels, counts = np.unique(labels, return_counts=True)
    gcc_label = unique_labels[np.argmax(counts)]
    gcc_indices = np.where(labels == gcc_label)[0]

    A_csr = A.tocsr()
    A_gcc = A_csr[gcc_indices, :][:, gcc_indices].tocoo()

    idx_to_node = {idx: name for name, idx in node_to_idx.items()}
    new_node_to_idx = {
        idx_to_node[old_idx]: new_idx for new_idx, old_idx in enumerate(gcc_indices)
    }

    gcc_fraction = len(gcc_indices) / n_original_nodes

    return A_gcc, new_node_to_idx, gcc_fraction


def calculate_sampled_shortest_path(A: coo_matrix, sample_size: int = 500) -> float:
    """
    Calculates the Average Shortest Path Length using a random sample of nodes.
    This prevents memory overload and massive computation times on 18k+ node graphs
    while maintaining statistical accuracy.
    """
    n_nodes = A.shape[0]

    # If network is smaller than sample size, just use all nodes
    actual_sample_size = min(sample_size, n_nodes)

    np.random.seed(42)
    sampled_indices = np.random.choice(n_nodes, size=actual_sample_size, replace=False)

    # Calculate shortest paths from ONLY the sampled nodes to all other nodes
    # unweighted=True forces simple hop-counting
    distances = shortest_path(
        csgraph=A, directed=False, indices=sampled_indices, unweighted=True
    )

    # Filter out infinite distances (disconnected nodes) and zeroes (distance to self)
    valid_distances = distances[(distances > 0) & (~np.isinf(distances))]

    if len(valid_distances) == 0:
        return np.nan

    return np.mean(valid_distances)


def calculate_modularity(A: coo_matrix, iterations: int = 10) -> float:
    """
    Calculates the maximum Modularity (Q) of the network.
    Because modularity maximization is highly degenerate (especially on PPIs),
    we run the algorithm multiple times with different random seeds and return
    the highest discovered Q score to ensure consistency.
    """
    # 1. Convert Scipy Sparse to igraph (Leiden requires igraph)
    sources, targets = A.nonzero()
    edges = list(zip(sources, targets))

    G_ig = ig.Graph(n=A.shape[0], edges=edges, directed=False)

    best_q = -1.0

    for i in range(iterations):
        # 2. Run Leiden iteratively
        partition = la.find_partition(G_ig, la.ModularityVertexPartition, seed=42 + i)

        current_q = partition.quality()
        if current_q > best_q:
            best_q = current_q

    return best_q
