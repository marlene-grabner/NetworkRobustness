"""
Community structure of the perturbed networks, one row per network and
algorithm:

  ari_baseline_all, ari_baseline_connected   mean ARI between the partitions
      of the perturbed network and those of its unperturbed network (all
      seeds x all seeds), on all nodes / on the connected nodes
  n_valid_baseline_pairs                     pairs in ari_baseline_connected
  ari_seeds_all, ari_seeds_connected         mean ARI between the seeds of the
      perturbed network itself
  the collapse statistics (similarity.collapse_statistics)

See similarity.py for the two node sets and the degenerate partitions.
"""

import numpy as np
import pandas as pd

from ..utils.perturbed_io import (
    iter_perturbed_networks,
    parse_perturbed_path,
    read_baseline_edges,
)
from .partitions import DEFAULT_ALGORITHMS, SEEDS, detect_partitions
from .similarity import collapse_statistics, mean_ari


def baseline_partitions(
    baseline_edgelists: list[str],
    node_to_idx: dict,
    algorithms=DEFAULT_ALGORITHMS,
    seeds=SEEDS,
) -> dict:
    """
    The partitions of the unperturbed networks: "{algorithm}_{replicate}" ->
    (n_seeds, n_nodes) label array. Can be saved with np.savez.
    """
    partitions = {}
    for replicate, edgelist in enumerate(baseline_edgelists):
        edges = read_baseline_edges(edgelist, node_to_idx)
        for algorithm in algorithms:
            partitions[f"{algorithm}_{replicate}"] = detect_partitions(
                edges, len(node_to_idx), algorithm, seeds
            )
    return partitions


def community_properties(labels: np.ndarray, baseline_labels=None) -> dict:
    """The result columns for the partitions (all seeds) of one network."""
    result = {"n_seeds": len(labels)}
    if baseline_labels is not None:
        (
            result["ari_baseline_all"],
            result["ari_baseline_connected"],
            result["n_valid_baseline_pairs"],
        ) = mean_ari(labels, baseline_labels)
    result["ari_seeds_all"], result["ari_seeds_connected"], _ = mean_ari(labels)
    result.update(collapse_statistics(labels))
    return result


def community_properties_of_file(
    path: str,
    node_to_idx: dict,
    baselines: dict,
    algorithms=DEFAULT_ALGORITHMS,
    seeds=SEEDS,
) -> pd.DataFrame:
    """
    Community detection on every network in one perturbed file. Columns:
    network, model, noise_type, target, action, level, replicate, repeat,
    algorithm and the result columns.

    :param node_to_idx: node index of the empirical network (load_node_index)
    :param baselines: baseline_partitions() of the unperturbed networks
    """
    info = parse_perturbed_path(path)
    rows = []
    for replicate, repeat, edges in iter_perturbed_networks(path, node_to_idx):
        for algorithm in algorithms:
            labels = detect_partitions(edges, len(node_to_idx), algorithm, seeds)
            rows.append(
                {
                    **info,
                    "replicate": replicate,
                    "repeat": repeat,
                    "algorithm": algorithm,
                    **community_properties(
                        labels, baselines[f"{algorithm}_{replicate}"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def community_properties_of_baselines(baselines: dict) -> pd.DataFrame:
    """The same columns (without the ARI to the baseline) for the unperturbed networks."""
    rows = []
    for key, labels in baselines.items():
        algorithm, replicate = key.rsplit("_", 1)
        rows.append(
            {
                "replicate": int(replicate),
                "algorithm": algorithm,
                **community_properties(labels),
            }
        )
    return pd.DataFrame(rows)
