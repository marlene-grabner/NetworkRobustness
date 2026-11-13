"""
Metrics on the score vectors of the seed expansion algorithms.

All score vectors are aligned to the same node index (the empirical network's,
utils.perturbed_io.load_node_index). Seeds carry -inf, so they are never part
of a top-k set and never count as positives or negatives. A node is "found"
(valid) if its score is > 0: reached by the random walk, added by DIAMOnD,
a neighbour of a seed for first neighbours.

Two questions per seed set:
  stability  does the perturbed network give the same top-k as the baseline?
             positives = baseline top-k
  recovery   does the algorithm find the hidden part of the module?
             positives = hidden module nodes (on the baseline and on the
             perturbed network alike)
"""

from __future__ import annotations

import numpy as np

K_LIST = [10, 20, 25, 50, 100]


def top_valid(scores: np.ndarray, k: int) -> np.ndarray:
    """The (at most) k best nodes with a score > 0, best first; ties by node index."""
    valid = np.flatnonzero(scores > 0)
    if len(valid) <= k:
        order = np.lexsort((valid, -scores[valid]))
    else:
        # only sort the k best (plus ties at the cut-off) instead of all nodes
        part = np.argpartition(-scores[valid], k - 1)
        cutoff = scores[valid[part[k - 1]]]
        valid = valid[scores[valid] >= cutoff]
        order = np.lexsort((valid, -scores[valid]))
    return valid[order][:k]


def n_tied_outside(scores: np.ndarray, top: np.ndarray, k: int) -> int:
    """
    Nodes outside the top-k with the same score as the k-th node: how many
    nodes the deterministic tie-break (lower node index first) decided against.
    0 means the top-k does not depend on the tie-break.
    """
    if len(top) < k:
        return 0
    kth = scores[top[k - 1]]
    return int(np.sum(scores == kth) - np.sum(scores[top[:k]] == kth))


def overlap_metrics(baseline_top: set, perturbed_top: set) -> dict:
    inter = len(baseline_top & perturbed_top)
    union = len(baseline_top | perturbed_top)
    # empty perturbed top-k (seeds cut off from the network): nothing recovered
    precision = inter / len(perturbed_top) if perturbed_top else 0.0
    recall = inter / len(baseline_top) if baseline_top else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    if np.isnan(recall):
        f1 = np.nan
    return {
        "jaccard": inter / union if union else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class RankedScores:
    """
    One score vector, sorted once, so AUROC and AUPRC can be computed for many
    sets of positives without sorting again. Gives the same values as
    sklearn's roc_auc_score and average_precision_score (ties included).
    Only nodes with a finite score (non-seeds) take part.
    """

    def __init__(self, scores: np.ndarray):
        self.scores = scores
        self.sorted = np.sort(scores[np.isfinite(scores)])
        self.n = len(self.sorted)

    def auroc_auprc(self, positives: np.ndarray) -> tuple[float, float]:
        s = self.scores[positives]
        s = s[np.isfinite(s)]
        n_pos = len(s)
        n_neg = self.n - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.nan, np.nan
        left = np.searchsorted(self.sorted, s, side="left")
        right = np.searchsorted(self.sorted, s, side="right")
        # AUROC = Mann-Whitney U with mid ranks
        mid_rank = left + (right - left + 1) / 2
        auroc = (mid_rank.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        # AUPRC = mean over positives of the precision at their score threshold
        s_sorted = np.sort(s)
        pos_at_or_above = n_pos - np.searchsorted(s_sorted, s, side="left")
        all_at_or_above = self.n - left
        auprc = float(np.mean(pos_at_or_above / all_at_or_above))
        return float(auroc), auprc


def ranking_summary(scores: np.ndarray, k_list=K_LIST) -> dict:
    """What is kept of a score vector: the top max(k) nodes and per-k tie counts."""
    k_max = max(k_list)
    top = top_valid(scores, k_max)
    return {
        "top": top,
        "n_valid": int(np.sum(scores > 0)),
        "n_tied_outside": {k: n_tied_outside(scores, top, k) for k in k_list},
    }


def recovery_metrics(
    scores: np.ndarray, ranked: RankedScores, top: np.ndarray, hidden: np.ndarray, k_list=K_LIST
) -> list[dict]:
    """
    Per k: share of the hidden nodes in the top-k (recall) and of the top-k
    that is hidden (precision). Same for all k: AUROC/AUPRC and the
    R-precision, the share of the hidden nodes in the top-|hidden|, which is
    comparable between seed sets with different numbers of hidden nodes.
    """
    auroc, auprc = ranked.auroc_auprc(hidden)
    hidden_set = set(hidden.tolist())
    top_r = top if len(top) >= len(hidden) else top_valid(scores, len(hidden))
    r_precision = len(hidden_set.intersection(top_r[: len(hidden)].tolist())) / len(hidden)
    rows = []
    for k in k_list:
        found = len(hidden_set.intersection(top[:k].tolist()))
        rows.append(
            {
                "k": k,
                "recovery_recall": found / len(hidden_set),
                "recovery_precision": found / k,
                "recovery_r_precision": r_precision,
                "recovery_auroc": auroc,
                "recovery_auprc": auprc,
            }
        )
    return rows


def compare_to_baseline(
    ranked: RankedScores, top: np.ndarray, baseline_top: np.ndarray, k_list=K_LIST
) -> list[dict]:
    """Per k: overlap of the top-k sets, AUROC/AUPRC of the perturbed scores for the baseline top-k."""
    rows = []
    for k in k_list:
        positives = baseline_top[:k]
        row = {"k": k, **overlap_metrics(set(positives.tolist()), set(top[:k].tolist()))}
        row["auroc"], row["auprc"] = ranked.auroc_auprc(positives)
        rows.append(row)
    return rows
