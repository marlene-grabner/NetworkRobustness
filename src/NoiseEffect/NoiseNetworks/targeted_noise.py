"""
Targeted noise: hub-, periphery- and betweenness-weighted edge addition and
removal. All weights are computed once on the baseline (no recalculation).
"""

import igraph as ig
import networkx as nx
import numpy as np


def _targetedAddition(g: nx.Graph, num_modify: int, target: str, graph_info: dict):
    """
    Adds non-edges whose two endpoints are each sampled by a node weight:
    the degree ('hubs') or the inverse degree ('periphery').
    """
    nodes = graph_info["nodes"]
    degrees = graph_info["degrees"]
    edges_to_add = set()

    max_possible_edges = (len(nodes) * (len(nodes) - 1)) // 2 - g.number_of_edges()
    if num_modify > max_possible_edges:
        raise ValueError("More edges to be added than available non-edges.")

    # Weights per NODE, not per edge. max(..., 1) gives degree-0 nodes the
    # same weight as degree-1 nodes, so isolated nodes can be reconnected.
    if target == "hubs":
        node_weights = np.array([max(degrees[n], 1) for n in nodes], dtype=float)
    elif target == "periphery":
        node_weights = np.array([1.0 / max(degrees[n], 1) for n in nodes], dtype=float)
    else:
        raise ValueError("Target must be 'hubs' or 'periphery'")
    node_probs = node_weights / node_weights.sum()

    while len(edges_to_add) < num_modify:
        # Sample in batches (2x what is still needed covers the rejections)
        batch_size = (num_modify - len(edges_to_add)) * 2
        u_batch = np.random.choice(nodes, size=batch_size, p=node_probs)
        v_batch = np.random.choice(nodes, size=batch_size, p=node_probs)

        for u, v in zip(u_batch, v_batch):
            if u != v and not g.has_edge(u, v):
                edges_to_add.add((u, v) if u < v else (v, u))
                if len(edges_to_add) == num_modify:
                    break

    return list(edges_to_add)


def _targetedRemoval(g: nx.Graph, num_modify: int, target: str, graph_info: dict):
    """
    Removes edges sampled without replacement by an edge weight: the product
    of the endpoint degrees ('hubs'), its inverse ('periphery') or the edge
    betweenness ('betweenness').
    """
    edges = graph_info["edges"]
    degrees = graph_info["degrees"]

    if target == "hubs":
        weights = np.array([degrees[u] * degrees[v] for u, v in edges], dtype=float)
    elif target == "periphery":
        # max(..., 1) prevents a division by zero if isolated nodes exist
        weights = np.array(
            [1.0 / (max(degrees[u], 1) * max(degrees[v], 1)) for u, v in edges],
            dtype=float,
        )
    elif target == "betweenness":
        weights = graph_info["edge_betweenness"]  # aligned with edges
    else:
        raise ValueError("Target must be 'hubs', 'periphery' or 'betweenness'")

    probabilities = weights / weights.sum()
    chosen_indices = np.random.choice(
        len(edges), size=num_modify, replace=False, p=probabilities
    )
    return [edges[i] for i in chosen_indices]


def _edgeBetweenness(g: nx.Graph, edges: list) -> np.ndarray:
    """
    Unnormalized edge betweenness of every baseline edge, in the order of
    edges. Uses igraph, because networkx is too slow on the large networks.
    """
    # igraph keeps the edge order of the input, so the result aligns with edges
    g_ig = ig.Graph(n=g.number_of_nodes(), edges=edges, directed=False)
    return np.array(g_ig.edge_betweenness(directed=False), dtype=float)
