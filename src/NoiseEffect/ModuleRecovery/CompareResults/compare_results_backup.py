import logging
import numpy as np
from .comparison_metrics_class import ComparisonMetrics
from ..module_result import ModuleResult

## new
from typing import Dict, Set, Optional
from dataclasses import dataclass

# Create the logging channel for this file
logger = logging.getLogger(__name__)


def compareResults(
    baselineResult_obj: ModuleResult, perturbedResult_obj: ModuleResult
) -> Dict[str, any]:
    """
    Compare baseline and perturbed module results.
    """
    if perturbedResult_obj.algorithm_type == "ranked":
        metrics = compareRankedResults(baselineResult_obj, perturbedResult_obj)
    else:
        metrics = compareSetResults(baselineResult_obj, perturbedResult_obj)

    # Convert to flat dictionary for CSV
    return _metricsToFlatDict(metrics)


def compareRankedResults(
    baselineResult_obj: ModuleResult, perturbedResult_obj: ModuleResult
) -> ComparisonMetrics:
    """
    Compare two ranked module results (RWR, DIAMOnD).
    """
    baseline_nodes = baselineResult_obj.nodes_ranked
    perturbed_nodes = perturbedResult_obj.nodes_ranked

    # Get full sets for overall comparison
    baseline_set = set(baseline_nodes.keys())
    perturbed_set = set(perturbed_nodes.keys())

    # --- Universal Set Metrics ---
    intersection = baseline_set & perturbed_set
    union = baseline_set | perturbed_set

    jaccard = len(intersection) / len(union) if union else 0.0
    overlap_coef = (
        len(intersection) / min(len(baseline_set), len(perturbed_set))
        if baseline_set and perturbed_set
        else 0.0
    )

    size_baseline = len(baseline_set)
    size_perturbed = len(perturbed_set)
    size_diff = size_perturbed - size_baseline
    size_ratio = size_perturbed / size_baseline if size_baseline > 0 else 0.0

    # --- Top-K Jaccard (at multiple K values) ---
    k_values = [10, 20, 50, 100, 200, 500]
    top_k_jaccard = {}

    for k in k_values:
        baseline_top_k = set(list(baseline_nodes.keys())[:k])
        perturbed_top_k = set(list(perturbed_nodes.keys())[:k])

        if baseline_top_k or perturbed_top_k:
            k_intersection = baseline_top_k & perturbed_top_k
            k_union = baseline_top_k | perturbed_top_k
            top_k_jaccard[k] = len(k_intersection) / len(k_union) if k_union else 0.0
        else:
            top_k_jaccard[k] = 0.0

    # --- Rank Correlation (for common nodes only) ---
    common_nodes = baseline_set & perturbed_set

    if len(common_nodes) > 1:
        # Get ranks for common nodes
        baseline_ranks = {node: rank for rank, node in enumerate(baseline_nodes.keys())}
        perturbed_ranks = {
            node: rank for rank, node in enumerate(perturbed_nodes.keys())
        }

        baseline_rank_array = np.array([baseline_ranks[node] for node in common_nodes])
        perturbed_rank_array = np.array(
            [perturbed_ranks[node] for node in common_nodes]
        )

        # Kendall Tau (range: -1 to 1, measures ordinal association)
        kendall_tau = _kendall_tau(baseline_rank_array, perturbed_rank_array)

        # Spearman (range: -1 to 1, Pearson on ranks)
        spearman_rho = _spearman_correlation(baseline_rank_array, perturbed_rank_array)
    else:
        kendall_tau = None
        spearman_rho = None

    # --- Weighted Jaccard (using scores) ---
    weighted_jaccard = _weighted_jaccard(baseline_nodes, perturbed_nodes)

    # --- Precision/Recall/F1 (treating baseline as "ground truth") ---
    precision = len(intersection) / size_perturbed if size_perturbed > 0 else 0.0
    recall = len(intersection) / size_baseline if size_baseline > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return ComparisonMetrics(
        jaccard=jaccard,
        overlap_coefficient=overlap_coef,
        module_size_baseline=size_baseline,
        module_size_perturbed=size_perturbed,
        size_difference=size_diff,
        size_ratio=size_ratio,
        top_k_jaccard=top_k_jaccard,
        kendall_tau=kendall_tau,
        spearman_rho=spearman_rho,
        weighted_jaccard=weighted_jaccard,
        precision=precision,
        recall=recall,
        f1=f1,
    )

    return ComparisonMetrics(
        jaccard=jaccard,
        overlap_coefficient=overlap_coef,
        module_size_baseline=size_baseline,
        module_size_perturbed=size_perturbed,
        size_difference=size_diff,
        size_ratio=size_ratio,
        top_k_jaccard=top_k_jaccard,
        kendall_tau=kendall_tau,
        spearman_rho=spearman_rho,
        weighted_jaccard=weighted_jaccard,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def compareSetResults(
    baselineResult_obj: ModuleResult, perturbedResult_obj: ModuleResult
) -> ComparisonMetrics:
    """
    Compare two set-based module results (DOMINO, 1st Neighbors).
    """
    baseline_set = baselineResult_obj.nodes_set
    perturbed_set = perturbedResult_obj.nodes_set

    # --- Set Metrics ---
    intersection = baseline_set & perturbed_set
    union = baseline_set | perturbed_set

    jaccard = len(intersection) / len(union) if union else 0.0
    overlap_coef = (
        len(intersection) / min(len(baseline_set), len(perturbed_set))
        if baseline_set and perturbed_set
        else 0.0
    )

    size_baseline = len(baseline_set)
    size_perturbed = len(perturbed_set)
    size_diff = size_perturbed - size_baseline
    size_ratio = size_perturbed / size_baseline if size_baseline > 0 else 0.0

    # --- Precision/Recall/F1 ---
    precision = len(intersection) / size_perturbed if size_perturbed > 0 else 0.0
    recall = len(intersection) / size_baseline if size_baseline > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return ComparisonMetrics(
        jaccard=jaccard,
        overlap_coefficient=overlap_coef,
        module_size_baseline=size_baseline,
        module_size_perturbed=size_perturbed,
        size_difference=size_diff,
        size_ratio=size_ratio,
        precision=precision,
        recall=recall,
        f1=f1,
        # Ranked-specific metrics are None for set-based
        top_k_jaccard=None,
        kendall_tau=None,
        spearman_rho=None,
        weighted_jaccard=None,
    )


def _metricsToFlatDict(metrics: ComparisonMetrics) -> Dict[str, any]:
    """
    Convert ComparisonMetrics to a flat dictionary suitable for CSV.
    Handles nested dictionaries (like top_k_jaccard) by flattening them.
    """
    flat_dict = {
        # Universal metrics (always present)
        "jaccard": metrics.jaccard,
        "overlap_coefficient": metrics.overlap_coefficient,
        "module_size_baseline": metrics.module_size_baseline,
        "module_size_perturbed": metrics.module_size_perturbed,
        "size_difference": metrics.size_difference,
        "size_ratio": metrics.size_ratio,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        # Ranked-specific (will be None for set-based)
        "kendall_tau": metrics.kendall_tau,
        "spearman_rho": metrics.spearman_rho,
        "weighted_jaccard": metrics.weighted_jaccard,
    }

    # Flatten top_k_jaccard dictionary
    if metrics.top_k_jaccard is not None:
        for k, value in metrics.top_k_jaccard.items():
            flat_dict[f"jaccard_top_{k}"] = value
    else:
        # For set-based algorithms, these columns will be None
        for k in [10, 20, 50, 100, 200, 500]:
            flat_dict[f"jaccard_top_{k}"] = None

    return flat_dict


# --- Helper functions for rank correlations ---


def _kendall_tau(ranks1: np.ndarray, ranks2: np.ndarray) -> float:
    """Calculate Kendall's Tau correlation coefficient."""
    from scipy.stats import kendalltau

    tau, _ = kendalltau(ranks1, ranks2)
    return tau if not np.isnan(tau) else 0.0


def _spearman_correlation(ranks1: np.ndarray, ranks2: np.ndarray) -> float:
    """Calculate Spearman's rank correlation coefficient."""
    from scipy.stats import spearmanr

    rho, _ = spearmanr(ranks1, ranks2)
    return rho if not np.isnan(rho) else 0.0


def _weighted_jaccard(dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
    """
    Calculate weighted Jaccard similarity using scores as weights.
    Useful for ranked results where scores matter, not just presence.
    """
    all_nodes = set(dict1.keys()) | set(dict2.keys())

    if not all_nodes:
        return 0.0

    numerator = 0.0
    denominator = 0.0

    for node in all_nodes:
        w1 = dict1.get(node, 0.0)
        w2 = dict2.get(node, 0.0)
        numerator += min(w1, w2)
        denominator += max(w1, w2)

    return numerator / denominator if denominator > 0 else 0.0


"""
Key Comparison Strategies by Algorithm Type

Ranked outputs (RWR, DIAMOnD, ROBUST):

AUPRC: Best metric for evaluating ranking quality
Top-K Jaccard/Precision/Recall: Take top-K nodes where K = size of baseline
Rank Biased Overlap (RBO): If you want to compare two rankings


Set outputs (DOMINO, 1st Neighbors):

Jaccard Index: Standard overlap metric
Precision/Recall/F1: Classification-style metrics
Size difference: Track if methods produce larger/smaller modules


Mixed comparisons (your typical case):

Baseline is usually ground truth (set)
Take Top-K from ranked methods to compare as sets
Keep AUPRC as additional metric for ranked methods
    """

'''

def compareResults(baselineResult_obj: ModuleResult, perturbedResult_obj: ModuleResult):
    # --- CASE 1: RANKED RESULTS (RWR, DIAMOnD) ---
    if perturbedResult_obj.algorithm_type == "ranked":
        metrics = compareRankedReasults(baselineResult_obj, perturbedResult_obj)
    # --- CASE 2: SET RESULTS (DOMINO, 1stNeighbors) ---
    else:
        metrics = compareSetResults(baselineResult_obj, perturbedResult_obj)


def compareRankedReasults(
    baselineResult_obj: ModuleResult, perturbedResult_obj: ModuleResult
):
    # --- K-Jaccard ---
    # --- Top-K Precision/Recall/F1 ---
    # --- Overall Precision-Recall Curve (AUPRC) ---
    pass


def compareSetResults(
    baselineResult_obj: ModuleResult, perturbedResult_obj: ModuleResult
):
    # --- Jaccard ---
    # --- Precision/Recall/F1 ---
    # --- Rank Biased Overlap (RBO) ---
    # --- Size Difference ---
    pass


def compareResults(baselineResult_obj: ModuleResult, perturbedResult_obj: ModuleResult):
    """
    baseline_module: ALWAYS a set/list of nodes (The Truth).
    recovered_module: Can be a Dict (Ranked) OR a Set/List (Unranked).
    """
    baseline_module = baselineResult_obj
    recovered_module = perturbedResult_obj
    baseline_set = baseline_module.as_set()
    k = len(baseline_set)  # The "Target Size" for Top-K

    metrics = {
        "jaccard": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "overlap_size": 0,
        "auprc": np.nan,
        "baseline_size": k,
        "recovered_size": recovered_module.size(),
    }

    # --- CASE 1: RANKED RESULTS (RWR, DIAMOnD) ---
    if recovered_module.algorithm_type == "ranked":
        # Calculate AUPRC using full ranking
        metrics["auprc"] = _calculate_auprc(baseline_set, recovered_module.nodes_ranked)

        # Calculate Top-K metrics (take top-k where k = baseline size)
        recovered_top_k = recovered_module.get_top_k(k)

    # --- CASE 2: SET RESULTS (DOMINO, 1stNeighbors) ---
    else:
        recovered_top_k = recovered_module.as_set()

    # Calculate overlap metrics
    intersection = len(baseline_set.intersection(recovered_top_k))
    union = len(baseline_set.union(recovered_top_k))

    metrics["jaccard"] = intersection / union if union > 0 else 0.0
    metrics["overlap_size"] = intersection
    metrics["precision"] = (
        intersection / len(recovered_top_k) if len(recovered_top_k) > 0 else 0.0
    )
    metrics["recall"] = intersection / k if k > 0 else 0.0

    # Calculate F1
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = (
            2
            * (metrics["precision"] * metrics["recall"])
            / (metrics["precision"] + metrics["recall"])
        )

    return metrics


def _calculate_auprc(true_positives: set, ranked_results: dict) -> float:
    """Calculate Area Under Precision-Recall Curve."""
    if not ranked_results or not true_positives:
        return 0.0

    ranked_nodes = list(ranked_results.keys())
    n_positives = len(true_positives)

    precisions = []
    recalls = []
    tp_count = 0

    for i, node in enumerate(ranked_nodes, 1):
        if node in true_positives:
            tp_count += 1

        precision = tp_count / i
        recall = tp_count / n_positives

        precisions.append(precision)
        recalls.append(recall)

    # Calculate area using trapezoidal rule
    auprc = np.trapz(precisions, recalls)
    return float(auprc)

'''

"""
def compareResults(algorithm: str, baseline_module: dict, recovered_module: dict):
    print("TODO: compareResults; what kind of comparison metrics are needed?")
    if algorithm == "1stNeighbors":
        print("Not implemented yet")
        results = []
    elif algorithm == "DIAMOnD":
        logger.warning("DIAMOnD is not implemented yet.")
        print("Not implemented yet")
        results = []
    elif algorithm == "DOMINO":
        print("Not implemented yet")
        results = []
    elif algorithm == "ROBUST":
        print("Not implemented yet")
        results = []
    elif algorithm == "ROBUST(bias_aware)":
        print("Not implemented yet")
        results = []
    elif algorithm == "RandomWalkWithRestart":
        print("x")
    else:
        raise ValueError(f"Unknown algorithm specified: {algorithm}")

    return results

"""
