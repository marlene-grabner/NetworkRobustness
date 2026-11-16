import numpy as np
import random
from .metrics import calculate_module_metrics, get_random_connected_subgraph


def generate_null_metrics(
    G, module_size, lcc_size, dist_matrix, node_to_idx, iterations=100
):
    nodes = list(G.nodes())
    nulls = {"lcc": [], "qs": [], "qlcc": [], "cond": [], "avg_sp": [], "sig_frac": []}

    for _ in range(iterations):
        random_nodes = set(random.sample(nodes, module_size))
        # Pass the matrix down
        lcc, qs, _, _, cond, avg_sp, sig_frac = calculate_module_metrics(
            G, random_nodes, dist_matrix, node_to_idx
        )

        nulls["lcc"].append(lcc)
        nulls["qs"].append(qs)
        nulls["cond"].append(cond)
        nulls["avg_sp"].append(avg_sp)
        nulls["sig_frac"].append(sig_frac)

        if lcc_size > 0:
            random_connected = get_random_connected_subgraph(G, lcc_size)
            _, _, q_lcc_null, _, _, _, _ = calculate_module_metrics(
                G, random_connected, dist_matrix, node_to_idx
            )
            nulls["qlcc"].append(q_lcc_null)
        else:
            nulls["qlcc"].append(0.0)

    return {k: (np.mean(v), np.std(v)) for k, v in nulls.items()}


def evaluate_network_modules(
    G, modules, network_metadata, dist_matrix, node_to_idx, iterations_null=100
):
    """
    Evaluates all modules on a SINGLE network and returns a list of dictionaries.
    """
    results = []

    def get_z(val, stat_tuple):
        mu, std = stat_tuple
        return (val - mu) / std if std > 0 else 0.0

    for disease, nodes in modules.items():
        valid_size = len(nodes.intersection(set(G.nodes())))
        if valid_size == 0:
            continue

        lcc, qs, q_lcc, lcc_size, cond, avg_sp, sig_frac = calculate_module_metrics(
            G, nodes, dist_matrix, node_to_idx
        )

        null_stats = generate_null_metrics(
            G,
            valid_size,
            lcc_size,
            dist_matrix,
            node_to_idx,
            iterations=iterations_null,
        )

        z_lcc = get_z(lcc, null_stats["lcc"])
        z_qs = get_z(qs, null_stats["qs"])
        z_qlcc = get_z(q_lcc, null_stats["qlcc"])
        z_cond = get_z(cond, null_stats["cond"])
        z_avg_sp = get_z(avg_sp, null_stats["avg_sp"])
        z_sig_frac = get_z(sig_frac, null_stats["sig_frac"])

        is_detectable = (
            (z_lcc >= 1.96)
            or (z_qs >= 1.96)
            or (z_qlcc >= 1.96)
            or (z_cond <= -1.96)
            or (z_avg_sp <= -1.96)
            or (z_sig_frac <= 1.96)
        )

        # Merge network info (noise level, etc.) with the module metrics
        row = {
            "disease": disease,
            "module_size": valid_size,
            "lcc_size": lcc_size,
            "lcc_fraction": lcc,
            "modularity_qs": qs,
            "modularity_qlcc": q_lcc,
            "conductance": cond,
            "avg_shortest_path": avg_sp,
            "significance_fraction": sig_frac,
            "z_score_lcc": z_lcc,
            "z_score_qs": z_qs,
            "z_score_qlcc": z_qlcc,
            "z_score_cond": z_cond,
            "z_score_avg_sp": z_avg_sp,
            "z_score_sig_frac": z_sig_frac,
            "is_detectable": is_detectable,
        }
        row.update(network_metadata)  # Adds perturbation_type, noise_level, etc.
        results.append(row)

    return results
