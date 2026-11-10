"""
Structure-aware noise: triadic closure (addition) and degree-preserving
rewiring. Both use the baseline structure only (no recalculation).
"""

import random

import networkx as nx
import numpy as np

# Give up if a sampler needs this many attempts per edge it still has to find
MAX_ATTEMPTS_PER_EDGE = 1000


def _sorted_edge(u, v):
    return (u, v) if u < v else (v, u)


############################################################
# Triadic closure, resource allocation type


def _countTriadicCandidates(g: nx.Graph) -> int:
    """Number of non-adjacent node pairs with at least one common neighbour."""
    A = nx.to_scipy_sparse_array(g, nodelist=range(g.number_of_nodes()), format="csr")
    A.data[:] = 1
    paths2 = (A @ A).tocsr()
    paths2.setdiag(0)
    paths2.eliminate_zeros()
    candidates = paths2 - paths2.multiply(A)  # drop pairs that are already edges
    candidates.eliminate_zeros()
    return candidates.nnz // 2


def _triadicClosureEdgesToAdd(g: nx.Graph, num_modify: int, graph_info: dict):
    """
    Closes open paths u - v - w, sampled so that a non-adjacent pair (u, w) is
    added with probability proportional to its resource allocation index
    sum_{v in common neighbours} 1 / k_v:

      1. pick an edge uniformly, and one of its endpoints as the centre v
         (the other endpoint is u)
      2. pick w uniformly from ALL neighbours of v
      3. reject if w == u, if u and w are already connected or already added

    Each path u - v - w is reached from both of its edges, so it is chosen with
    probability proportional to 1 / k_v.
    """
    if num_modify > graph_info["n_triadic_candidates"]:
        raise ValueError(
            f"{num_modify} edges to add, but only {graph_info['n_triadic_candidates']} "
            "non-adjacent pairs have a common neighbour."
        )
    edges = graph_info["edges"]
    neighbours = graph_info["neighbours"]
    edges_to_add = set()
    attempts = 0

    while len(edges_to_add) < num_modify:
        # Sample in batches (2x what is still needed covers the rejections)
        batch_size = (num_modify - len(edges_to_add)) * 2
        attempts += batch_size
        if attempts > MAX_ATTEMPTS_PER_EDGE * num_modify:
            raise RuntimeError("Triadic closure sampling did not converge.")

        edge_idx = np.random.randint(0, len(edges), size=batch_size)
        centre_is_first = np.random.randint(0, 2, size=batch_size).astype(bool)
        uniform = np.random.random(size=batch_size)

        for i, first, x in zip(edge_idx, centre_is_first, uniform):
            a, b = edges[i]
            v, u = (a, b) if first else (b, a)
            w = neighbours[v][int(x * len(neighbours[v]))]
            if w != u and not g.has_edge(u, w):
                edges_to_add.add(_sorted_edge(u, w))
                if len(edges_to_add) == num_modify:
                    break

    return list(edges_to_add)


############################################################
# Degree-preserving rewiring


def _rewireEdges(g: nx.Graph, num_modify: int, graph_info: dict) -> nx.Graph:
    """
    Double-edge swaps until num_modify of the original edges are gone (one more
    at most, because a swap can replace two original edges at once):
    (a, b), (c, d) -> (a, d), (c, b). Swaps creating a self-loop or an existing
    edge are rejected. Every node keeps its degree and the edge count stays
    the same.

    Not every noise level is reachable: a fully rewired network still shares
    some edges with the original by chance (as the configuration model does),
    so very high levels raise an error instead of running forever.
    """
    edges = [_sorted_edge(u, v) for u, v in graph_info["edges"]]
    original = set(edges)  # sorted too, so membership checks match
    edge_set = set(edges)
    n_edges = len(edges)
    n_original_left = n_edges
    target_left = n_edges - num_modify
    attempts = 0

    while n_original_left > target_left:
        attempts += 1
        if attempts > MAX_ATTEMPTS_PER_EDGE * n_edges:
            raise RuntimeError(
                f"Rewiring did not reach {num_modify} replaced edges; "
                "the noise level is probably higher than reachable."
            )
        i, j = random.randrange(n_edges), random.randrange(n_edges)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if random.random() < 0.5:  # both ways of swapping are possible
            c, d = d, c
        if a == d or c == b:  # self-loop
            continue
        new1, new2 = _sorted_edge(a, d), _sorted_edge(c, b)
        if new1 in edge_set or new2 in edge_set:  # multi-edge
            continue

        for old in (edges[i], edges[j]):
            edge_set.remove(old)
            n_original_left -= old in original
        for new in (new1, new2):
            edge_set.add(new)
            n_original_left += new in original
        edges[i], edges[j] = new1, new2

    G_rewired = nx.Graph()
    G_rewired.add_nodes_from(g.nodes())  # keeps isolated nodes
    G_rewired.add_edges_from(edges)
    return G_rewired
