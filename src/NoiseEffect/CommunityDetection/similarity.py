"""
Comparing partitions: adjusted Rand index (ARI) and collapse statistics.

A partition is a label array over all baseline nodes; nodes without edges
have the label -1 (see partitions.detect_partitions).

Every ARI is calculated on two node sets:
  all        all baseline nodes; a node without edges is its own community,
             so nodes that lose all their edges count as lost structure
  connected  only nodes with at least one edge in both networks

Degenerate partitions (one single community, or every node alone):
  compared to a baseline: ARI 0, unless the baseline partition is degenerate
  too; then the pair is excluded (NaN)
  between seeds of one network: ARI 0 if one is degenerate, excluded if both are
"""

import itertools

import numpy as np

NEAR_COLLAPSE = 0.95  # largest community holds at least this fraction of the nodes


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI of two label arrays (same result as sklearn, without its overhead)."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    pairs = lambda counts: (counts * (counts - 1) / 2).sum()  # noqa: E731
    together_both = pairs(np.unique(a * (b.max() + 1) + b, return_counts=True)[1])
    together_a, together_b = pairs(np.bincount(a)), pairs(np.bincount(b))
    expected = together_a * together_b / (len(a) * (len(a) - 1) / 2)
    maximum = (together_a + together_b) / 2
    if maximum == expected:  # both degenerate in the same way
        return 1.0
    return float((together_both - expected) / (maximum - expected))


def is_degenerate(labels: np.ndarray) -> bool:
    """One single community, or every node in its own community."""
    n_communities = len(np.unique(labels))
    return n_communities <= 1 or n_communities == len(labels)


def ari_all_and_connected(a: np.ndarray, b: np.ndarray, b_is_baseline: bool):
    """(ARI on all nodes, ARI on the connected nodes) of two partitions."""
    connected = (a != -1) & (b != -1)
    return (
        _ari(_isolated_as_singletons(a), _isolated_as_singletons(b), b_is_baseline),
        _ari(a[connected], b[connected], b_is_baseline),
    )


def _ari(a, b, b_is_baseline) -> float:
    degenerate_a, degenerate_b = is_degenerate(a), is_degenerate(b)
    if degenerate_b if b_is_baseline else (degenerate_a and degenerate_b):
        return np.nan
    if degenerate_a or degenerate_b:
        return 0.0
    return adjusted_rand_index(a, b)


def _isolated_as_singletons(labels: np.ndarray) -> np.ndarray:
    """Gives every node without edges (-1) a community of its own."""
    labels = labels.copy()
    isolated = labels == -1
    labels[isolated] = labels.max() + 1 + np.arange(isolated.sum())
    return labels


def mean_ari(label_matrix: np.ndarray, baseline_matrix: np.ndarray | None = None):
    """
    Mean ARI over the valid pairs, as (all nodes, connected nodes, valid pairs):
    between all seeds of label_matrix, or, with a baseline_matrix, between
    every seed of label_matrix and every seed of the baseline.
    """
    if baseline_matrix is None:
        pairs = itertools.combinations(label_matrix, 2)
    else:
        pairs = itertools.product(label_matrix, baseline_matrix)
    scores = np.array(
        [ari_all_and_connected(a, b, baseline_matrix is not None) for a, b in pairs]
    )
    valid = ~np.isnan(scores)
    with np.errstate(invalid="ignore"):
        means = np.nansum(scores, axis=0) / valid.sum(axis=0)
    return float(means[0]), float(means[1]), int(valid[:, 1].sum())


def collapse_statistics(label_matrix: np.ndarray) -> dict:
    """
    Community statistics over the seeds of one network, on the nodes with
    edges (nodes without edges are in no community):

      n_communities            mean number of communities
      largest_community_frac   mean fraction of the nodes in the largest community
      n_collapsed              seeds with one single community
      n_near_collapsed         seeds where the largest community holds at least
                               95% of the nodes, but not all
      n_all_singletons         seeds where every node is alone
    """
    n_communities, largest = [], []
    for labels in label_matrix:
        sizes = np.unique(labels[labels != -1], return_counts=True)[1]
        n_communities.append(len(sizes))
        largest.append(sizes.max() / sizes.sum() if len(sizes) else np.nan)
    n_communities, largest = np.array(n_communities), np.array(largest)
    n_connected = int((label_matrix[0] != -1).sum())
    return {
        "n_connected_nodes": n_connected,
        "n_communities": float(n_communities.mean()),
        "largest_community_frac": float(np.mean(largest)),
        "n_collapsed": int((n_communities == 1).sum()),
        "n_near_collapsed": int(
            ((largest >= NEAR_COLLAPSE) & (n_communities > 1)).sum()
        ),
        "n_all_singletons": int(
            ((n_communities == n_connected) & (n_communities > 1)).sum()
        ),
    }
