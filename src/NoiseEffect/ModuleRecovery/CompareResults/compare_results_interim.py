from ..module_result import ModuleResult


def compareResults(
    baselineResult_obj: ModuleResult, perturbedResult_obj: ModuleResult
) -> dict[str, any]:
    """
    Compare baseline and perturbed module results.
    """
    if perturbedResult_obj.algorithm_type == "ranked":
        metrics = compareRankedResults(baselineResult_obj, perturbedResult_obj)
    else:
        metrics = compareSetResults(baselineResult_obj, perturbedResult_obj)

    # Convert to flat dictionary for CSV
    return metrics


def compareRankedResults(baselineResult_obj, perturbedResult_obj):
    top_100_baseline_nodes = set(list(baselineResult_obj.nodes_ranked.keys())[:100])
    top_100_perturbed_nodes = set(list(perturbedResult_obj.nodes_ranked.keys())[:100])
    jaccard_100 = len(
        top_100_baseline_nodes.intersection(top_100_perturbed_nodes)
    ) / len(top_100_baseline_nodes.union(top_100_perturbed_nodes))
    return {"top_100_jaccard": jaccard_100}


def compareSetResults(baselineResult_obj, perturbedResult_obj):
    jaccard = len(
        baselineResult_obj.nodes_set.intersection(perturbedResult_obj.nodes_set)
    ) / len(baselineResult_obj.nodes_set.union(perturbedResult_obj.nodes_set))
    return jaccard
