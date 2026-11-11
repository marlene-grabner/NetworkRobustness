"""
Global properties of the perturbed networks, one row per network.

  metric            columns
  gcc               gcc, num_singletons
  efficiency        efficiency (sampled, see metrics.global_efficiency)
  transitivity      transitivity
  assortativity     assortativity
  kcore             max_core, max_core_frac
  eigenvalue        largest_eigenvalue
  centrality_rank   {degree|pagerank}_rank_corr, _top1_overlap, _top10_overlap
                    (vs the unperturbed network)
"""

import pandas as pd

from . import metrics as m
from .perturbed_io import (
    iter_perturbed_networks,
    parse_perturbed_path,
    read_baseline_edges,
)

METRICS = {
    "gcc": m.gcc_and_singletons,
    "efficiency": m.global_efficiency,
    "transitivity": m.transitivity,
    "assortativity": m.assortativity,
    "kcore": m.kcore,
    "eigenvalue": m.largest_eigenvalue,
    "centrality_rank": m.centrality_rank,  # needs the unperturbed network
}


def global_properties(
    edges, n_nodes: int, metrics=tuple(METRICS), reference=None
) -> dict:
    """
    The chosen metrics of one network.

    :param edges: (m, 2) integer array, node indices in 0..n_nodes-1
    :param n_nodes: number of baseline nodes (also those without edges)
    :param reference: metrics.centralities() of the unperturbed network, only
        needed for centrality_rank
    """
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    g = m.build_graph(edges, n_nodes)
    result = {}
    for metric in metrics:
        if metric == "centrality_rank":
            if reference is None:
                raise ValueError("centrality_rank needs the reference centralities.")
            result.update(m.centrality_rank(g, reference))
        else:
            result.update(METRICS[metric](g))
    return result


def global_properties_of_file(
    path: str, node_to_idx: dict, baseline_edgelists: list[str], metrics=tuple(METRICS)
) -> pd.DataFrame:
    """
    The chosen metrics of every network in one perturbed file. Columns:
    network, model, noise_type, target, action, level, replicate, repeat and
    the metric columns.

    :param node_to_idx: node index of the empirical network (load_node_index)
    :param baseline_edgelists: the unperturbed edgelist of every replicate, in
        the order used to generate the file (one entry for an empirical network)
    """
    info = parse_perturbed_path(path)
    n_nodes = len(node_to_idx)
    references = {}  # replicate -> centralities of its unperturbed network
    rows = []
    for replicate, repeat, edges in iter_perturbed_networks(path, node_to_idx):
        if "centrality_rank" in metrics and replicate not in references:
            baseline = read_baseline_edges(baseline_edgelists[replicate], node_to_idx)
            references[replicate] = m.centralities(m.build_graph(baseline, n_nodes))
        properties = global_properties(
            edges, n_nodes, metrics, references.get(replicate)
        )
        rows.append({**info, "replicate": replicate, "repeat": repeat, **properties})
    return pd.DataFrame(rows)


def global_properties_of_baseline(
    path_to_edgelist: str, node_to_idx: dict, metrics=tuple(METRICS)
) -> dict:
    """The chosen metrics of an unperturbed network (centrality ranks are 1)."""
    edges = read_baseline_edges(path_to_edgelist, node_to_idx)
    n_nodes = len(node_to_idx)
    reference = m.centralities(m.build_graph(edges, n_nodes))
    return global_properties(edges, n_nodes, metrics, reference)
