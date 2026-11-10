"""Random noise: uniform edge addition and removal."""

import random

import networkx as nx
import numpy as np


def _randomEdgesToAdd(g: nx.Graph, num_modify: int, graph_info: dict):
    """Adds non-edges between uniformly chosen node pairs (rejection sampling)."""
    nodes = graph_info["nodes"]
    num_nodes = len(nodes)
    edges_to_add = set()

    max_possible_edges = (num_nodes * (num_nodes - 1)) // 2 - g.number_of_edges()
    if num_modify > max_possible_edges:
        raise ValueError("More edges to be added than available non-edges.")

    while len(edges_to_add) < num_modify:
        # Sample in batches (2x what is still needed covers the rejections)
        batch_size = (num_modify - len(edges_to_add)) * 2
        u_indices = np.random.randint(0, num_nodes, size=batch_size)
        v_indices = np.random.randint(0, num_nodes, size=batch_size)

        for u_idx, v_idx in zip(u_indices, v_indices):
            if u_idx != v_idx:  # no self-loops
                u, v = nodes[u_idx], nodes[v_idx]
                if not g.has_edge(u, v):
                    # sorted, so (u,v) and (v,u) are the same set entry
                    edges_to_add.add((u, v) if u < v else (v, u))
                    if len(edges_to_add) == num_modify:
                        break

    return list(edges_to_add)


def _randomEdgesToRemove(g: nx.Graph, num_modify: int, graph_info: dict):
    """Removes uniformly sampled edges."""
    return random.sample(graph_info["edges"], num_modify)
