import itertools
import numpy as np
import pandas as pd
import igraph as ig
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from NoiseEffect.CommunityDetection.detection_algorithms import (
    leidenAlgorithmPartioning, louvainPartioning,
    infomapAlgorithmPartioning, labelPropagationPartitioning
)
from NoiseEffect.CommunityDetection.utils import convertPartitionToLabels, mask_isolated_nodes

def run_algorithm(ig_graph, algo, seeds):
    if algo == "leiden": return leidenAlgorithmPartioning(ig_graph, seeds, n_iterations=2)
    if algo == "louvain": return louvainPartioning(ig_graph, seeds)
    if algo == "infomap": return infomapAlgorithmPartioning(ig_graph, seeds, n_iterations=1)
    if algo == "label_propagation": return labelPropagationPartitioning(ig_graph, seeds)
    raise ValueError(f"Unknown algorithm: {algo}")

def safe_ari(labels_a, labels_b):
    # Nodes labeled -1 (degree-0 - see mask_isolated_nodes) are excluded first.
    # Within-network pairs share the same -1 pattern (same graph on both sides);
    # cross-network pairs mask on the intersection, since a node isolated in only
    # one of the two networks still has a real label on the other side.
    valid_mask = (labels_a != -1) & (labels_b != -1)
    a_filtered = labels_a[valid_mask]
    b_filtered = labels_b[valid_mask]

    if len(a_filtered) == 0: return np.nan
    u_a, u_b = np.unique(a_filtered), np.unique(b_filtered)
    if len(u_a) == 1 or len(u_b) == 1: return np.nan
    if len(u_a) == len(a_filtered) or len(u_b) == len(b_filtered): return np.nan
    try: return float(adjusted_rand_score(a_filtered, b_filtered))
    except: return np.nan

def calculate_aris(label_matrix, baseline_labels=None):
    # Internal pairwise
    internal_scores = [safe_ari(label_matrix[i], label_matrix[j])
                       for i, j in itertools.combinations(range(len(label_matrix)), 2)]
    arr_int = np.array(internal_scores, dtype=float)
    valid_int = arr_int[~np.isnan(arr_int)]
    within_mean = float(np.mean(valid_int)) if len(valid_int) > 0 else np.nan
    within_std = float(np.std(valid_int)) if len(valid_int) > 0 else np.nan

    # Cross pairwise
    vs_base_mean, vs_base_std = np.nan, np.nan
    if baseline_labels is not None:
        cross_scores = [safe_ari(label_matrix[i], baseline_labels[j])
                        for i in range(len(label_matrix)) for j in range(len(baseline_labels))]
        arr_cross = np.array(cross_scores, dtype=float)
        valid_cross = arr_cross[~np.isnan(arr_cross)]
        vs_base_mean = float(np.mean(valid_cross)) if len(valid_cross) > 0 else np.nan
        vs_base_std = float(np.std(valid_cross)) if len(valid_cross) > 0 else np.nan

    return within_mean, within_std, vs_base_mean, vs_base_std

def _process_one_network(repeat_id, df_edges, n_nodes, algo, seeds, baseline_labels):
    """Internal worker function for a single graph instance."""
    # Build over the full n_nodes universe (not just nodes that appear in this
    # repeat's edgelist) so degree-0 nodes exist as vertices and community
    # detection runs on the complete graph.
    edges = list(zip(df_edges["source"], df_edges["target"]))
    g = ig.Graph(n=n_nodes, edges=edges)
    has_edge = np.array(g.degree()) > 0

    partitions = run_algorithm(g, algo, seeds)
    label_matrix = np.stack([convertPartitionToLabels(partitions[s], n_nodes) for s in seeds])
    # Stamp -1 back onto this repeat's degree-0 nodes so safe_ari's -1 filtering
    # masks them out, regardless of the trivial singleton label the algorithm gave them.
    label_matrix = mask_isolated_nodes(label_matrix, has_edge)

    w_mean, w_std, vb_mean, vb_std = calculate_aris(label_matrix, baseline_labels)

    return {
        "repeat_id": repeat_id,
        "within_ari_mean": w_mean,
        "within_ari_std": w_std,
        "vs_baseline_ari_mean": vb_mean,
        "vs_baseline_ari_std": vb_std,
        "mean_n_communities": float(np.mean([len(partitions[s]) for s in seeds])),
        "std_n_communities": float(np.std([len(partitions[s]) for s in seeds])),
    }

def evaluate_network_repeats(df_pert, n_nodes, algo, seeds, baseline_labels, n_jobs=1):
    """
    Takes a mapped Parquet dataframe containing multiple repeats,
    processes them in parallel, and returns a list of dictionaries.
    """
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_one_network)(
            repeat_id, group[['source', 'target']], n_nodes, algo, seeds, baseline_labels
        )
        for repeat_id, group in df_pert.groupby("repeat")
    )
    return results
