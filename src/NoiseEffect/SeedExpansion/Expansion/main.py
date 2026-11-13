"""
Seed expansion on the unperturbed and perturbed networks.

For every seed set (SyntheticSeeds; seeds + hidden module nodes) and algorithm
the score vector is computed and reduced to
  stability  perturbed top-k vs baseline top-k: jaccard, precision, recall,
             f1, auroc, auprc (positives = baseline top-k)
  recovery   share of the hidden module nodes in the top-k
             (recovery_recall, recovery_precision), in the top-|hidden|
             (recovery_r_precision) and recovery_auroc / recovery_auprc, on
             the baseline and on every perturbed network
  ties       n_tied_outside: nodes outside the top-k with the same score as
             the k-th node, i.e. how often the deterministic tie-break decided
  n_valid    nodes with a score > 0 (reached / added / seed neighbours)

The baseline keeps only the top max(K_LIST) nodes per seed set and algorithm;
that is all the stability metrics need.
"""

import numpy as np
import pandas as pd
from scipy import sparse

from ...utils.perturbed_io import (
    iter_perturbed_networks,
    parse_perturbed_path,
    read_baseline_edges,
)
from .comparing import (
    K_LIST,
    RankedScores,
    compare_to_baseline,
    ranking_summary,
    recovery_metrics,
)
from .seed_expansion_on_graph import run_algorithm

DEFAULT_ALGORITHMS = ["rwr_row", "rwr_sym", "diamond", "first_neighbors"]


def edges_to_adjacency(edges: np.ndarray, n_nodes: int) -> sparse.csr_matrix:
    """Symmetric 0/1 CSR adjacency from an (m, 2) integer edge array."""
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adj = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()
    adj.data[:] = 1.0  # duplicate edges stay unweighted
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def seed_sets_of_unit(seed_table: pd.DataFrame, network: str, model: str, node_to_idx: dict) -> dict:
    """replicate -> list of (set_id, seed_type, seed indices, hidden indices)"""
    rows = seed_table[(seed_table.network == network) & (seed_table.model == model)]
    if rows.empty:
        raise ValueError(f"No seed sets for {network}/{model}")

    def to_idx(labels):
        return np.array([node_to_idx[n] for n in labels.split(";")], dtype=np.int64)

    sets = {}
    for row in rows.itertuples():
        sets.setdefault(int(row.replicate), []).append(
            (int(row.set_id), row.seed_type, to_idx(row.seed_nodes), to_idx(row.hidden_nodes))
        )
    return sets


def _results_of_network(adj, seed_sets, algorithms, k_list, baseline_tops=None):
    """
    Rows for one network (all seed sets and algorithms) and the top lists
    (algorithm, set_id) -> top nodes. Without baseline_tops (= the baseline
    itself) the stability columns are left out.
    """
    rows, tops = [], {}
    for set_id, seed_type, seeds, hidden in seed_sets:
        for algorithm in algorithms:
            scores = run_algorithm(algorithm, adj, seeds)
            summary = ranking_summary(scores, k_list)
            ranked = RankedScores(scores)
            tops[(algorithm, set_id)] = summary["top"]
            per_k = recovery_metrics(scores, ranked, summary["top"], hidden, k_list)
            if baseline_tops is not None:
                stability = compare_to_baseline(
                    ranked, summary["top"], baseline_tops[(algorithm, set_id)], k_list
                )
                per_k = [{**r, **s} for r, s in zip(per_k, stability)]
            for row in per_k:
                rows.append(
                    {
                        "set_id": set_id,
                        "seed_type": seed_type,
                        "algorithm": algorithm,
                        **row,
                        "n_tied_outside": summary["n_tied_outside"][row["k"]],
                        "n_valid": summary["n_valid"],
                    }
                )
    return rows, tops


def baseline_seed_expansion(
    baseline_edgelists: list[str],
    node_to_idx: dict,
    seed_sets: dict,
    algorithms=DEFAULT_ALGORITHMS,
    k_list=K_LIST,
) -> tuple[pd.DataFrame, dict]:
    """
    Seed expansion on the unperturbed networks (one per replicate).

    :return: the result rows (replicate, set_id, seed_type, algorithm, k, recovery_*,
        n_tied_outside, n_valid) and the top lists
        "{algorithm}|{replicate}|{set_id}" -> top max(k_list) node indices,
        which can be saved with np.savez
    """
    rows, tops = [], {}
    for replicate, edgelist in enumerate(baseline_edgelists):
        adj = edges_to_adjacency(read_baseline_edges(edgelist, node_to_idx), len(node_to_idx))
        network_rows, network_tops = _results_of_network(
            adj, seed_sets[replicate], algorithms, k_list
        )
        rows += [{"replicate": replicate, **r} for r in network_rows]
        for (algorithm, set_id), top in network_tops.items():
            tops[f"{algorithm}|{replicate}|{set_id}"] = top
    return pd.DataFrame(rows), tops


def seed_expansion_of_file(
    path: str,
    node_to_idx: dict,
    seed_sets: dict,
    baseline_tops: dict,
    algorithms=DEFAULT_ALGORITHMS,
    k_list=K_LIST,
) -> pd.DataFrame:
    """
    Seed expansion on every network in one perturbed file. Columns: network,
    model, noise_type, target, action, level, replicate, repeat, set_id,
    seed_type, algorithm, k and the result columns.

    :param node_to_idx: node index of the empirical network (load_node_index)
    :param seed_sets: seed_sets_of_unit() of this network and model
    :param baseline_tops: the top lists of baseline_seed_expansion()
    """
    info = parse_perturbed_path(path)
    rows = []
    for replicate, repeat, edges in iter_perturbed_networks(path, node_to_idx):
        adj = edges_to_adjacency(edges, len(node_to_idx))
        tops = {
            (algorithm, set_id): baseline_tops[f"{algorithm}|{replicate}|{set_id}"]
            for set_id, *_ in seed_sets[replicate]
            for algorithm in algorithms
        }
        network_rows, _ = _results_of_network(
            adj, seed_sets[replicate], algorithms, k_list, baseline_tops=tops
        )
        rows += [{**info, "replicate": replicate, "repeat": repeat, **r} for r in network_rows]
    return pd.DataFrame(rows)
