import networkx as nx
import numpy as np
import pandas as pd
import random

def generate_single_bfs_seed(G: nx.Graph, start_node, num_seeds: int = 20) -> list:
    """
    Traverses the graph using BFS from a start node until it collects 
    the requested number of unique local neighborhood nodes.
    """
    # nx.bfs_tree returns nodes in strict layer-by-layer BFS order
    bfs_nodes = list(nx.bfs_tree(G, source=start_node))
    
    if len(bfs_nodes) < num_seeds:
        raise ValueError(f"Graph component too small to yield {num_seeds} seeds.")
        
    return bfs_nodes[:num_seeds]


def get_prioritized_candidates_by_percentile(G: nx.Graph, target_percentile: float) -> tuple[float, list]:
    """
    Ranks ALL nodes in the graph by their proximity to the target degree percentile.
    Returns the target degree value and the full list of nodes ordered from closest to furthest.
    """
    if not (0.0 <= target_percentile <= 1.0):
        raise ValueError("Percentile must be between 0.0 and 1.0")

    all_nodes = list(G.nodes())
    degrees = np.array([G.degree(n) for n in all_nodes])
    
    # Calculate the absolute target degree value at this percentile
    target_degree = np.percentile(degrees, target_percentile * 100)
    
    # Calculate absolute distance of every node's degree from our target degree
    node_distances = [
        (node, abs(G.degree(node) - target_degree)) 
        for node in all_nodes
    ]
    
    # Sort nodes by distance (closest to target degree first)
    # If distances are equal, sort by node ID to keep sorting stable
    node_distances.sort(key=lambda x: (x[1], x[0]))
    
    # Extract the full list of nodes, now perfectly ordered by percentile proximity
    sorted_candidates = [node for node, dist in node_distances]
    
    return target_degree, sorted_candidates