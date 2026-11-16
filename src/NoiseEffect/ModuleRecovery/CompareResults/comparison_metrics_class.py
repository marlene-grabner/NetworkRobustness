from typing import Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class ComparisonMetrics:
    """
    Standardized comparison metrics that work across both ranked and set-based algorithms.
    All metrics are in [0, 1] range where applicable for easy comparison.
    """

    # Ranked-specific metrics (None for set-based)
    top_k_jaccard: Optional[Dict[int, float]] = None  # {k: jaccard_score}
    kendall_tau: Optional[float] = None
    spearman_rho: Optional[float] = None
    weighted_jaccard: Optional[float] = None
    rank_correlation_top_k: Optional[Dict[int, float]] = None

    # Set-specific or general
    jaccard: float
    overlap_coefficient: float
    module_size_baseline: int
    module_size_perturbed: int
    size_difference: int
    size_ratio: float  # perturbed / baseline
    precision: Optional[float] = None  # perturbed relative to baseline
    recall: Optional[float] = None
    f1: Optional[float] = None
