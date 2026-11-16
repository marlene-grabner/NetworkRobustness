import numpy as np
from scipy.optimize import linear_sum_assignment


def applyRecall(df, baselines, k=100):
    """
    Function called by main.py.
    Searches for the baseline corresponding to each row's seed_id and applies Jaccard comparison.
    """

    def row_processor(row):
        seed_id = row.get("seed_id")
        if seed_id not in baselines:
            return None

        baseline_data = baselines[seed_id]
        perturbed_data = row["results"]
        result_type = row.get("result_type")

        return _computeScore(baseline_data, perturbed_data, result_type, k)

    return df.apply(row_processor, axis=1)


def _computeScore(baseline, perturbed, result_type, k):
    """
    Checks data type and selects the appropriate logic for each case.
    """
    # Standardize
    base_set = set()
    pert_set = set()

    # Handle the case that an algorithm returned null
    if result_type == "null_result":
        return 0.0

    # Handle the case of multiple modules (e.g. DOMINO)
    if result_type == "multi_module":
        return _recallForMultiModule(baseline, perturbed)

    # Handle ranked RWR results
    elif result_type == "rwr":
        base_set = _getTopkSet(baseline, k)
        pert_set = _getTopkSet(perturbed, k)

    # Handle DIAMOnD results
    elif result_type == "DIAMOnD":
        base_set = set(node for node, score in baseline)
        pert_set = set(node for node, score in perturbed)

    # Handle single sets (1st Neighbors)
    elif result_type == "single_module":
        base_set = set(baseline) if baseline else set()
        pert_set = set(perturbed) if perturbed else set()

    # If results are empty return 0 if only the perturbed is empty, 1 if both are empty
    elif result_type == "empty":
        if len(baseline) == 0:
            print(
                "Warning: Both baseline and perturbed results are empty. Returning Recall of 1.0."
            )
            return 1.0
        else:
            print(
                "Warning: Perturbed results are empty but baseline is not. Returning Recall of 0.0."
            )
            return 0.0

    # Prevent ZeroDivisionError
    if len(base_set) == 0:
        if len(pert_set) == 0:
            print(
                "Warning: Both baseline and perturbed results are empty, despite not being classified as empty. Returning Recall of 1.0."
            )
            return 1.0
        return 0.0

    return len(base_set.intersection(pert_set)) / len(base_set)


########## List of Sets case ##########


def _recallForMultiModule(baseline_modules, perturbed_modules):
    """
    Computes the optimal one to one matching score between lists of sets.
    """
    # 1. Handle the empty case
    n_base = len(baseline_modules)
    n_pert = len(perturbed_modules)

    if n_base == 0 and n_pert == 0:
        return "both empty"  # Both empty = perfect match
    if n_base == 0 or n_pert == 0:
        return "one empty"  # One empty = failure

    # Build Cost Matrix
    cost_matrix = np.zeros((n_base, n_pert))

    for i, b_mod in enumerate(baseline_modules):
        for j, p_mod in enumerate(perturbed_modules):
            # negative Recall because linear_sum_assignment minimizes cost
            cost_matrix[i, j] = -recall_set(b_mod, p_mod)

    # 3. Solve Assignment Problem (The Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. Sum the Recall scores of the optimal pairs
    # the sum is negated back to positive
    total_recall = -cost_matrix[row_ind, col_ind].sum()

    # 5. Normalizing
    # Dividing by max(n_base, n_pert) penalizes:
    # - Missing modules (unmatched baseline rows)
    # - Extra noise modules (unmatched perturbed columns)
    return total_recall / max(n_base, n_pert)


########## Helper for List of Sets case ##########
def recall_set(s1, s2):
    """Standard Recall for two sets"""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    inter = len(s1.intersection(s2))
    return inter / len(s1)


########## Helper for RWR case ##########


def _getTopkSet(ranked_data, k):
    """
    Extracts top K keys from a dictionary based on values.
    """
    if not isinstance(ranked_data, dict):
        return set()
    # Sort by score descending and take top k keys
    top_keys = sorted(ranked_data, key=ranked_data.get, reverse=True)[:k]
    return set(top_keys)
