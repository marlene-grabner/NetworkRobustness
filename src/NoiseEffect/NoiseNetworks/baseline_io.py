"""Loading the baseline networks."""

import networkx as nx


def _loadBaseline(path_to_edgelist: str, path_to_nodelist: str | None = None):
    """
    Loads the baseline network and relabels it to integer nodes 0..n-1.
    Returns the integer-labelled graph and both label mappings.

    :param path_to_edgelist: Path to the baseline edgelist: a headerless csv,
        one comma-separated edge per line
    :param path_to_nodelist: Optional comma-separated file of degree-0 nodes that
        are missing from the edgelist
    """
    g_raw = nx.read_edgelist(path_to_edgelist, delimiter=",")
    if path_to_nodelist is not None:
        with open(path_to_nodelist) as f:
            content = f.read().strip()
        all_nodes = content.split(",") if content else []
        g_raw.add_nodes_from(n.strip() for n in all_nodes if n.strip())

    if not all(isinstance(n, str) for n in g_raw.nodes()):
        raise TypeError("All node labels must be strings; got mixed/non-string types.")
    print(
        f"Loaded {g_raw.number_of_nodes()} nodes and {g_raw.number_of_edges()} edges from the edgelist."
    )

    # Sorted, so the mapping is always the same
    original_node_labels = sorted(g_raw.nodes())
    idx_to_node = dict(enumerate(original_node_labels))
    node_to_idx = {label: i for i, label in idx_to_node.items()}
    g = nx.relabel_nodes(g_raw, node_to_idx)
    return g, idx_to_node, node_to_idx

