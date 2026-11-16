import networkx as nx
import numpy as np
import random
from scipy.stats import hypergeom


def get_random_connected_subgraph(G, target_size, max_attempts=100):
    """
    Samples a random connected subgraph of exactly `target_size`.
    Uses a boundary expansion approach starting from a random node.
    """
    if target_size <= 0:
        return set()
    if target_size == 1:
        return {random.choice(list(G.nodes()))}

    nodes_list = list(G.nodes())

    for _ in range(max_attempts):
        start_node = random.choice(nodes_list)
        subgraph_nodes = {start_node}
        boundary_nodes = set(G.neighbors(start_node))

        # Grow the subgraph until we hit the target size
        while len(subgraph_nodes) < target_size and boundary_nodes:
            next_node = random.choice(list(boundary_nodes))
            subgraph_nodes.add(next_node)
            boundary_nodes.remove(next_node)

            # Add new neighbors to the boundary, excluding already visited ones
            new_neighbors = set(G.neighbors(next_node)) - subgraph_nodes
            boundary_nodes.update(new_neighbors)

        if len(subgraph_nodes) == target_size:
            return subgraph_nodes

    # Fallback: If we fail 100 times (e.g., in a highly fragmented graph),
    # fall back to standard random sampling to prevent infinite loops.
    print(
        f"Warning: Failed to sample a connected subgraph of size {target_size} after {max_attempts} attempts. Falling back to random sampling."
    )
    return set(random.sample(nodes_list, target_size))


def calculate_module_metrics(G, module_nodes, dist_matrix, node_to_idx):
    """Calculates absolute topological properties of a module within a graph."""
    valid_nodes = module_nodes.intersection(set(G.nodes()))
    if not valid_nodes:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0, 1.0

    subgraph = G.subgraph(valid_nodes)
    E_total = G.number_of_edges()
    N_total = G.number_of_nodes()

    if E_total == 0 or N_total < 2:
        return 0.0, 0.0, 0.0, 0, 0.0, 0.0, 1.0

    # 1. Modularity (q_s)
    e_s = subgraph.number_of_edges()
    k_s = sum(dict(G.degree(valid_nodes)).values())
    q_s = (e_s / E_total) - ((k_s / (2 * E_total)) ** 2)

    # 2. LCC fraction and Modularity
    if len(subgraph) == 0:
        lcc_fraction, q_lcc, lcc_size = 0.0, 0.0, 0
    else:
        lcc_nodes = max(nx.connected_components(subgraph), key=len)
        lcc_size = len(lcc_nodes)
        lcc_fraction = lcc_size / len(valid_nodes)
        lcc_subgraph = G.subgraph(lcc_nodes)
        q_lcc = (lcc_subgraph.number_of_edges() / E_total) - (
            (sum(dict(G.degree(lcc_nodes)).values()) / (2 * E_total)) ** 2
        )

    # 3. Conductance
    try:
        cond = nx.conductance(G, valid_nodes)
    except nx.NetworkXError:
        cond = 0.0

    # 4. Average Shortest Path in G
    distances = []
    valid_list = list(valid_nodes)

    for i in range(len(valid_list)):
        for j in range(i + 1, len(valid_list)):
            idx_u = node_to_idx[valid_list[i]]
            idx_v = node_to_idx[valid_list[j]]

            d = dist_matrix[idx_u, idx_v]
            if not np.isinf(d):
                distances.append(d)

    avg_sp = np.mean(distances) if distances else 0.0

    # 5. Median Hypergeometric p-value
    p_values = []
    S_size = len(valid_nodes) - 1
    pop_size = N_total - 1

    for u in valid_nodes:
        k_u = G.degree(u)
        k_u_S = sum(1 for neighbor in G.neighbors(u) if neighbor in valid_nodes)
        p_val = hypergeom.sf(k_u_S - 1, pop_size, S_size, k_u) if k_u_S > 0 else 1.0
        p_values.append(p_val)

    sig_fraction = sum(1 for p in p_values if p < 0.05) / len(valid_nodes)

    return lcc_fraction, q_s, q_lcc, lcc_size, cond, avg_sp, sig_fraction
