import os
import numpy as np
import igraph as ig
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)


def load_isolated_nodes(csv_path: str) -> set:
    """
    Loads degree-0 node IDs from the sidecar CSV next to a baseline edgelist
    (e.g. 'foo_sbm.csv' -> 'foo_sbm_isolated_nodes.csv'), if one exists.

    Null model generation (ER/SBM) can produce nodes with no edges, which can't
    be represented in an edgelist and are instead stored in this sidecar file.
    """
    sidecar = os.path.splitext(csv_path)[0] + "_isolated_nodes.csv"
    if not os.path.exists(sidecar):
        return set()
    with open(sidecar) as f:
        content = f.read().strip()
    return {n.strip() for n in content.split(",") if n.strip()}


def build_full_graph(df, isolated_nodes=None):
    """
    Builds an undirected igraph.Graph over the full node universe (edgelist nodes
    plus any isolated_nodes), so degree-0 nodes exist as vertices instead of being
    silently dropped by only ever reading them off of an edgelist.

    Returns:
        g: igraph.Graph with one vertex per node in the universe.
        node_order: list of original node IDs; node_order[i] is the ID of vertex i.
        has_edge: boolean np.ndarray of length len(node_order), True where that
            vertex has at least one edge (i.e. degree > 0) in this graph.
    """
    isolated_nodes = isolated_nodes or set()
    node_order = sorted(set(df["source"]) | set(df["target"]) | isolated_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(node_order)}
    edges = list(zip(df["source"].map(node_to_idx), df["target"].map(node_to_idx)))
    g = ig.Graph(n=len(node_order), edges=edges)
    has_edge = np.array(g.degree()) > 0
    return g, node_order, has_edge


def mask_isolated_nodes(label_matrix, has_edge):
    """
    Forces every degree-0 node's label back to -1 (the "unassigned" sentinel),
    regardless of what community detection assigned it.

    Community detection runs on the complete graph, including degree-0 nodes -
    but every algorithm except Infomap places a degree-0 node into a trivial
    singleton community of its own rather than leaving it unassigned. Since ARI
    comparisons need to ignore these nodes entirely, this restores the -1
    convention uniformly across algorithms right where the labels are built, so
    every downstream comparison can just filter on -1 without needing to know
    which nodes were isolated.
    """
    label_matrix = np.array(label_matrix, copy=True)
    label_matrix[..., ~has_edge] = -1
    return label_matrix


def convertPartitionToLabels(partition, num_nodes):
    """
    Converts a partition (list of sets of node IDs) to a labels array.

    This version infers the total number of nodes by finding the maximum
    node ID in the partition. It assumes nodes are indexed from 0.
    """
    # Handle the edge case of an empty or invalid partition
    if not partition or not any(partition):
        return np.array([], dtype=int)

    # 1. Create the labels array. Using np.full with -1 is often safer
    #    to make it obvious if a node was missed.
    labels = np.full(num_nodes, -1, dtype=int)

    # 2. Populate the array with cluster IDs.
    for cluster_id, community in enumerate(partition):
        for node in community:
            labels[node] = cluster_id

    return labels


def getMetrics(clustering_1, clustering_2):
    """
    Computes Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI)
    between two cluster labelings.

    Nodes labeled -1 in either clustering (degree-0 in that network - see
    mask_isolated_nodes) are excluded first: within-network comparisons end up
    masking on that network's own isolated nodes since both sides share the
    same -1 pattern, while cross-network comparisons mask on the intersection,
    since a node isolated in only one of the two networks still has a real
    label on the other side.
    """
    clustering_1 = np.asarray(clustering_1)
    clustering_2 = np.asarray(clustering_2)
    valid_mask = (clustering_1 != -1) & (clustering_2 != -1)
    clustering_1 = clustering_1[valid_mask]
    clustering_2 = clustering_2[valid_mask]

    if len(clustering_1) == 0:
        return {
            "status": "no_overlap",
            "num_clusters_1": 0,
            "num_clusters_2": 0,
            "ari": np.nan,
            "ami": np.nan,
        }

    n_clustering_1 = len(np.unique(clustering_1))
    n_clustering_2 = len(np.unique(clustering_2))

    # Handle there being only one cluster in either parition
    if n_clustering_1 == 1 and n_clustering_2 == 1:
        return {
            "status": "trivial_one_cluster",
            "num_clusters_1": n_clustering_1,
            "num_clusters_2": n_clustering_2,
            "ari": np.nan,
            "ami": np.nan,
        }

    # Handle all nodes being singletons in either partition
    if n_clustering_1 == len(clustering_1) or n_clustering_2 == len(clustering_2):
        return {
            "status": "trivial_all_singletons",
            "num_clusters_1": n_clustering_1,
            "num_clusters_2": n_clustering_2,
            "ari": np.nan,
            "ami": np.nan,
        }

    try:
        # Calculate metrics
        ari = adjusted_rand_score(labels_true=clustering_1, labels_pred=clustering_2)
        #ami = adjusted_mutual_info_score(
        #    labels_true=clustering_1, labels_pred=clustering_2
        #)
        return {
            "status": "success",
            "num_clusters_1": n_clustering_1,
            "num_clusters_2": n_clustering_2,
            "ari": ari,
            #"ami": ami,
            "ami": np.nan,  # Temporarily set AMI to NaN
        }
    except Exception as e:
        return {
            "status": "error",
            "num_clusters_1": n_clustering_1,
            "num_clusters_2": n_clustering_2,
            "ari": np.nan,
            "ami": np.nan,
        }
