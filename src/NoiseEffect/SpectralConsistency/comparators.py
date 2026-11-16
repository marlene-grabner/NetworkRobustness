import numpy as np
from typing import Dict
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import norm


def calculate_string_based_overlap(
    vec_orig: np.ndarray,
    vec_pert: np.ndarray,
    dict_orig: dict,
    dict_pert: dict,
    k: int = 200,
) -> float:
    """
    Calculates Jaccard overlap using the actual biological names of the nodes.
    This safely compares vectors of different sizes (e.g., when edge deletion
    shrinks the GCC) by matching string identities instead of raw matrix indices.
    """
    idx_to_node_orig = {v: k_name for k_name, v in dict_orig.items()}
    idx_to_node_pert = {v: k_name for k_name, v in dict_pert.items()}

    top_k_idx_orig = np.argsort(np.abs(vec_orig))[-k:]
    top_k_idx_pert = np.argsort(np.abs(vec_pert))[-k:]

    top_k_names_orig = set(idx_to_node_orig[idx] for idx in top_k_idx_orig)
    top_k_names_pert = set(idx_to_node_pert[idx] for idx in top_k_idx_pert)

    intersection = len(top_k_names_orig.intersection(top_k_names_pert))
    union = len(top_k_names_orig.union(top_k_names_pert))

    return intersection / union if union > 0 else 0.0


def calculate_davis_kahan_bounds(
    L_orig: coo_matrix,
    L_pert: coo_matrix,
    vals_orig: np.ndarray,
    vecs_orig: np.ndarray,
    vecs_pert: np.ndarray,
) -> Dict[str, float]:
    """
    Calculates the pure spectral perturbation metrics.
    WARNING: Matrices must be strictly identical in shape. Do not use on
    networks that have undergone edge deletion/GCC filtering.
    """
    # 1. The Force of the Noise (Spectral Norm of the difference)
    Delta_L = L_pert - L_orig
    norm_delta_L = norm(Delta_L, ord=2)

    # 2. The Structural Integrity (Spectral Gap)
    # Using index 1 (Fiedler value) and index 2
    lambda_2 = vals_orig[1]
    lambda_3 = vals_orig[2]
    spectral_gap = abs(lambda_3 - lambda_2)

    # 3. The Predicted Damage (Theoretical Davis-Kahan Bound)
    if spectral_gap > 0:
        theoretical_bound = norm_delta_L / spectral_gap
    else:
        theoretical_bound = np.inf

    # 4. The Actual Damage (Empirical angular shift of the Fiedler vectors)
    v2_orig = vecs_orig[:, 1]
    v2_pert = vecs_pert[:, 1]

    dot_product = np.abs(np.dot(v2_orig, v2_pert))
    dot_product = np.clip(dot_product, 0.0, 1.0)
    empirical_sin_theta = np.sin(np.arccos(dot_product))

    return {
        "Norm_Delta_L": norm_delta_L,
        "Orig_Spectral_Gap": spectral_gap,
        "DK_Theoretical_Bound": theoretical_bound,
        "Empirical_Shift_SinTheta": empirical_sin_theta,
    }
