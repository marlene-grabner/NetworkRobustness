import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def applyAUC(df, baselines, universe, k=100, metric_type="auroc"):
    """
    Applies AUC calculation strictly to ranked results.
    metric_type can be "auroc" or "auprc".
    """
    universe_list = list(universe)

    def row_processor(row):
        seed_id = row.get("seed_id")
        if seed_id not in baselines:
            return None

        baseline_data = baselines[seed_id]
        perturbed_data = row["results"]
        result_type = row.get("result_type")

        return _computeAUC(
            baseline_data, perturbed_data, result_type, universe_list, k, metric_type
        )

    return df.apply(row_processor, axis=1)


def _computeAUC(baseline, perturbed, result_type, universe, k, metric_type):
    """
    Calculates AUROC or AUPRC only for ranked algorithms. Returns None for flat sets.
    """
    # 1. Reject unranked result types immediately
    if result_type not in ["rwr", "DIAMOnD"]:
        return None

    # 2. Define Ground Truth (y_true) from the Baseline
    gt_set = set()
    if result_type == "rwr":
        gt_set = _getTopkSet(baseline, k)
    elif result_type == "DIAMOnD":
        gt_set = set(node for node, score in baseline)

    # If baseline is completely empty, area under the curve is undefined
    if len(gt_set) == 0:
        return None

    # 3. Define Predicted Scores (y_score) from the Perturbed data
    score_dict = {}
    if result_type == "rwr":
        # RWR values are already probabilities
        score_dict = perturbed
    elif result_type == "DIAMOnD":
        # Convert iteration rank to a pseudo-probability score (1st node = highest score)
        n_pert = len(perturbed)
        score_dict = {node: (n_pert - i) for i, (node, _) in enumerate(perturbed)}

    # 4. Build the arrays matching the full network universe
    y_true = np.array([1 if n in gt_set else 0 for n in universe])
    y_score = np.array([score_dict.get(n, 0.0) for n in universe])

    # 5. Calculate the chosen AUC metric
    try:
        if metric_type == "auprc":
            # average_precision_score is scikit-learn's function for AUPRC
            return average_precision_score(y_true, y_score)
        else:
            return roc_auc_score(y_true, y_score)

    except ValueError:
        # Triggers if y_true somehow only has one class
        return None


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
