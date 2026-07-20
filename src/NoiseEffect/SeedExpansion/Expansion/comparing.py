"""
Metrics comparing a perturbed-network score vector against the baseline
score vector for the same network/algorithm/seeds.

All functions work on dense numpy score vectors aligned to the SAME node
index (see io_helper.NodeIndex) -- this is why baseline and perturbed graphs
must share one index. Seed nodes are expected to carry -inf scores (as all
algorithms in src/algorithms already produce) so they never contaminate
top-k sets or positive/negative labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def topk_set_valid(scores: np.ndarray, k: int) -> set:
    """
    Returns the set of top-k indices, strictly excluding zero or -inf scores.
    If the requested k exceeds the number of valid scores, it safely caps at the valid length.
    """
    valid_count = int(np.sum(scores > 0))
    k_actual = min(k, valid_count)
    if k_actual == 0:
        return set()
    
    order = np.argsort(-scores, kind="stable")
    return set(order[:k_actual].tolist())


def topk_overlap_metrics(baseline_scores: np.ndarray, perturbed_scores: np.ndarray,
                          k: int) -> dict:
    a = set(topk_set_valid(baseline_scores, k))
    b = set(topk_set_valid(perturbed_scores, k))
    inter = len(a & b)
    union = len(a | b)

    jaccard = inter / union if union else np.nan
    precision = inter / len(b) if b else np.nan   # of the perturbed top-k, fraction also in baseline top-k
    recall = inter / len(a) if a else np.nan       # of the baseline top-k, fraction recovered
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) > 0 else np.nan)

    return {"k": k, "jaccard": jaccard, "precision": precision, "recall": recall, "f1": f1}


def auc_metrics(baseline_scores: np.ndarray, perturbed_scores: np.ndarray, k: int) -> dict:
    # Eligible nodes: exclude seeds (which are -inf in both arrays)
    eligible = np.isfinite(baseline_scores) & np.isfinite(perturbed_scores)
    if eligible.sum() < 2:
        return {"auroc": np.nan, "auprc": np.nan}

    pos_idx = topk_set_valid(baseline_scores, k)
    
    # If no true positives exist, or ALL eligible nodes are true positives, AUC is undefined
    if len(pos_idx) == 0 or len(pos_idx) == eligible.sum():
        return {"auroc": np.nan, "auprc": np.nan}

    eligible_indices = np.flatnonzero(eligible)
    y_true = np.array([1 if i in pos_idx else 0 for i in eligible_indices])
    y_score = perturbed_scores[eligible]

    try:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
    except ValueError:
        auroc, auprc = np.nan, np.nan

    return {"auroc": auroc, "auprc": auprc}


def compare_rankings(baseline_scores: np.ndarray, perturbed_scores: np.ndarray, k_list: list[int]) -> list[dict]:
    # 1. Count nodes with strict signal (excluding seeds at -inf and unreached at 0)
    n_valid_base = int(np.sum(baseline_scores > 0))
    n_valid_pert = int(np.sum(perturbed_scores > 0))
    
    # 2. Identify the exact rank where scores drop to zero/unadded
    rank_zero_base = n_valid_base + 1
    rank_zero_pert = n_valid_pert + 1
    
    # 3. The shortest valid list determines the dynamic cap
    k_shortest = min(n_valid_base, n_valid_pert)
    
    # 4. Merge requested k's with the shortest k, and deduplicate
    eval_ks = sorted(list(set(k_list + [k_shortest])))
    
    rows = []
    for k in eval_ks:
        # Skip evaluation if k=0 (e.g., if a graph is completely disconnected)
        if k == 0:
            continue
            
        row = topk_overlap_metrics(baseline_scores, perturbed_scores, k)
        row.update(auc_metrics(baseline_scores, perturbed_scores, k))
        
        # 5. Inject metadata columns into the final DataFrame
        row["rank_zero_base"] = rank_zero_base
        row["rank_zero_pert"] = rank_zero_pert
        row["is_shortest_k"] = (k == k_shortest)
        
        rows.append(row)
        
    return rows
