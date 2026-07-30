"""
Generates and saves baseline community detection partitions for network graphs.
Maps string-based node IDs to contiguous 0-based integers for C-backend compatibility,
computes community partitions across multiple random seeds, and saves the resulting 
label matrices to .npz files for downstream stability comparisons.
"""

import os
import igraph as ig
import numpy as np
import pandas as pd
from typing import Dict, Any, List


from NoiseEffect.CommunityDetection.detection_algorithms import (
    leidenAlgorithmPartioning,
    infomapAlgorithmPartioning,
    louvainPartioning,
    labelPropagationPartitioning,
)


def extract_node_mapping(df: pd.DataFrame) -> tuple[Dict[str, int], List[str]]:
    """
    Extracts unique nodes from an edgelist and maps them to contiguous 0-based integers.
    
    Args:
        df: Pandas DataFrame containing 'source' and 'target' columns.
        
    Returns:
        A tuple containing:
        - Dictionary mapping original string IDs to integer indices.
        - List of the original string IDs (where list index == integer mapped ID).
    """
    unique_nodes = sorted(list(set(df['source']) | set(df['target'])))
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    return node_to_idx, unique_nodes


def run_community_detection(ig_base: ig.Graph, algorithm: str, seeds: List[int], parameters: Dict[str, Any]) -> Dict[int, List[set]]:
    """
    Routes the graph to the specified community detection algorithm.
    """
    if algorithm == "leiden":
        n_iterations = parameters.get("n_iterations", 2)
        return leidenAlgorithmPartioning(ig_base, seeds, n_iterations=n_iterations)
        
    elif algorithm == "louvain":
        return louvainPartioning(ig_base, seeds)
        
    elif algorithm == "label_propagation":
        return labelPropagationPartitioning(ig_base, seeds)
        
    elif algorithm == "infomap":
        n_iterations = parameters.get("n_iterations", 20)
        return infomapAlgorithmPartioning(ig_base, seeds, n_iterations=n_iterations)
        
    else:
        raise ValueError(f"Unknown algorithm specified: {algorithm}")


def build_label_matrix(partitions: Dict[int, List[set]], seeds: List[int], num_nodes: int) -> np.ndarray:
    """
    Converts a dictionary of community sets into a dense 2D Numpy array of cluster labels.
    
    Args:
        partitions: Dict mapping seed -> List of communities (where each community is a set of node IDs).
        seeds: List of random seeds used.
        num_nodes: Total number of unique nodes in the network.
        
    Returns:
        np.ndarray of shape (num_seeds, num_nodes) containing cluster assignments.
    """
    labels = []
    for s in seeds:
        # Initialize with -1 to easily catch unassigned nodes
        lab_array = np.full(num_nodes, -1, dtype=int)
        for cluster_id, comm in enumerate(partitions[s]):
            for node in comm:
                lab_array[node] = cluster_id
        labels.append(lab_array)
        
    return np.stack(labels)


def generate_baseline_npz(
    algorithm: str, 
    baseline_csvs: Dict[str, str], 
    out_dir: str, 
    num_repeats: int = 20, 
    parameters: Dict[str, Any] = None
) -> None:
    """
    Main pipeline to generate and save baseline community partitions.
    
    Args:
        algorithm: The detection algorithm to use ('leiden', 'louvain', 'infomap', 'label_propagation').
        baseline_csvs: Dictionary mapping network names to their CSV filepaths.
        out_dir: Directory to save the resulting .npz files.
        num_repeats: Number of random seeds/repeats to execute per network.
        parameters: Optional hyperparameters for specific algorithms.
    """
    if parameters is None:
        parameters = {}
        
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(1, num_repeats + 1))
    
    for net_name, csv_path in baseline_csvs.items():
        print(f"Generating baseline for {net_name} using {algorithm}...")
        
        # 1. Load data safely as strings
        df = pd.read_csv(csv_path, names=['source', 'target'], dtype=str)
        
        # 2. Extract strict mapping and align the dataframe
        node_to_idx, unique_nodes = extract_node_mapping(df)
        df['source'] = df['source'].map(node_to_idx)
        df['target'] = df['target'].map(node_to_idx)
        
        # 3. Build igraph object
        ig_base = ig.Graph.DataFrame(df, directed=False)
        
        # 4. Execute Community Detection
        partitions = run_community_detection(ig_base, algorithm, seeds, parameters)
        
        # 5. Convert partitions to a 2D dense mathematical matrix
        labels_matrix = build_label_matrix(partitions, seeds, len(unique_nodes))
        
        # 6. Save matrix alongside the coordinate mapping (node_order)
        out_path = os.path.join(out_dir, f"{net_name}_{algorithm}.npz")
        np.savez(out_path, labels=labels_matrix, node_order=unique_nodes)
        print(f"Successfully saved {out_path}")


if __name__ == "__main__":
    baselines = {
        "ppi": "data/baseline_networks/chloe_ppi_lcc_2026_02_23.csv",
        "astro": "data/baseline_networks/ca-AstroPh_gcc.csv",
        "power":"data/baseline_networks/western_us_power_grid.csv",
        "wiki": "data/baseline_networks/wiki-Vote_gcc.csv",
        "ppi_er": "data/baseline_networks/null_models/chloe_ppi_erdos_renyi.csv",
        "ppi_conf": "data/baseline_networks/null_models/chloe_ppi_configuration_model.csv",
        "ppi_sbm": "data/baseline_networks/null_models/chloe_ppi_sbm.csv",
        "astro_er": "data/baseline_networks/null_models/ca-AstroPh_erdos_renyi.csv",
        "astro_conf": "data/baseline_networks/null_models/ca-AstroPh_configuration_model.csv",
        "astro_sbm": "data/baseline_networks/null_models/ca-AstroPh_sbm.csv",
        "power_er": "data/baseline_networks/null_models/western_us_power_grid_erdos_renyi.csv",
        "power_conf": "data/baseline_networks/null_models/western_us_power_grid_configuration_model.csv",
        "power_sbm": "data/baseline_networks/null_models/western_us_power_grid_sbm.csv",
        "wiki_er": "data/baseline_networks/null_models/wiki-Vote_erdos_renyi.csv",
        "wiki_conf": "data/baseline_networks/null_models/wiki-Vote_configuration_model.csv",
        "wiki_sbm": "data/baseline_networks/null_models/wiki-Vote_sbm.csv"
    }

    for algorithm in ["leiden", "louvain", "infomap", "label_propagation"]:
        print(f"Generating baselines for algorithm: {algorithm}")
        generate_baseline_npz(
            algorithm=algorithm, 
            baseline_csvs=baselines, 
            out_dir=f"./outputs/local_structure/baselines/{algorithm}/",
            num_repeats=20
        )

    

