import igraph as ig
import networkx as nx
import itertools
import numpy as np
from NoiseEffect.CommunityDetection.utils import convertPartitionToLabels, getMetrics, mask_isolated_nodes
from NoiseEffect.CommunityDetection.detection_algorithms import (
    leidenAlgorithmPartioning,
    infomapAlgorithmPartioning,
    louvainPartioning,
    labelPropagationPartitioning,
)


def benchmarkBaselineStabilityAlgorithm(ig_graph: ig.Graph, list_of_seeds: list[int], algorithm: str, parameters: dict = {}):
    """
    Runs a community detection algorithm multiple times to benchmark its stability.

    Args:
        ig_graph (igraph.Graph): The input graph.
        list_of_seeds (list[int]): A list of random seeds to run the algorithm with.
        algorithm (str): The name of the algorithm to use ('leiden', 'infomap').
        parameters (dict, optional): Extra parameters for the algorithm,
                                     e.g., {'n_iterations': 5}. Defaults to {}.

    Returns:
        list[dict]: A list of metric dictionaries, comparing all pairs
                    of runs. Each dict contains 'ari', 'ami', etc.
    """

    num_nodes = ig_graph.vcount()

    if algorithm == "leiden":
        n_iterations = parameters.get(
            "n_iterations", 2
        )  # Get specified n_iterations or use default chosed by leidenalg package (which is 2)
        partitions = leidenAlgorithmPartioning(
            ig_graph, list_of_seeds, n_iterations=n_iterations
        )
    elif algorithm == "louvain":
        partitions = louvainPartioning(ig_graph, list_of_seeds)
    elif algorithm == "label_propagation":
        partitions = labelPropagationPartitioning(ig_graph, list_of_seeds)
    elif algorithm == "sbm":
        raise NotImplementedError(
            "SBM partitioning not implemented in benchmarkAlgorithm."
        )
    elif algorithm == "infomap":
        n_iterations = parameters.get(
            "n_iterations", 1
        )  # Get specified n_iterations or use default value for infomap package which is 1
        partitions = infomapAlgorithmPartioning(
            ig_graph, list_of_seeds, n_iterations=n_iterations
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    labels = np.stack([
        convertPartitionToLabels(partition, num_nodes) for partition in partitions.values()
    ])
    # Community detection runs over the complete graph (including degree-0
    # nodes), which get placed in trivial singleton communities of their own;
    # stamp -1 back onto them so getMetrics' -1 filtering excludes them.
    has_edge = np.array(ig_graph.degree()) > 0
    labels = mask_isolated_nodes(labels, has_edge)

    all_ordered_pairs = itertools.combinations(labels, 2)

    results = []
    for pair in all_ordered_pairs:
        result = getMetrics(*pair)
        results.append(result)
    return results
