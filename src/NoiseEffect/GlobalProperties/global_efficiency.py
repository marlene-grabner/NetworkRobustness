import os
import numpy as np
import pandas as pd
import igraph as ig


def load_baseline_node_index(baseline_path: str, sep: str = ',') -> tuple[dict, int]:
    """
    Build a stable node -> integer index mapping for a baseline edgelist.

    Isolated nodes carry no edges and are therefore absent from the edgelist
    itself; they are pulled in from an optional sidecar file named
    '<baseline_path without extension>_isolated_nodes.csv' (a single line of
    comma-separated node ids, no header), if one exists next to the baseline
    file.

    :param baseline_path: Path to the baseline edgelist (two columns, no header)
    :param sep: Field separator of the edgelist file
    :return: (node_to_idx, n_nodes)
    """
    df_base = pd.read_csv(
        baseline_path, sep=sep, header=None, names=['source', 'target'], dtype=str
    )
    nodes = set(df_base['source']) | set(df_base['target'])

    sidecar = baseline_path.rsplit('.', 1)[0] + '_isolated_nodes.csv'
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            content = f.read().strip()
        if content:
            nodes |= {n.strip() for n in content.split(',') if n.strip()}

    node_to_idx = {n: i for i, n in enumerate(sorted(nodes))}
    return node_to_idx, len(node_to_idx)


def build_graph(edge_df: pd.DataFrame, node_to_idx: dict, n_nodes: int) -> ig.Graph:
    """
    Build an undirected igraph.Graph on n_nodes vertices (indexed 0..n_nodes-1,
    matching node_to_idx) from an edge dataframe with 'source'/'target' columns.

    Nodes that have no edges in edge_df (isolated nodes, including any not
    present in a perturbed edgelist that removed all of their edges) still
    end up in the graph as zero-degree vertices, since the graph is sized by
    n_nodes rather than by the nodes actually appearing in edge_df.
    """
    u = edge_df['source'].astype(str).map(node_to_idx)
    v = edge_df['target'].astype(str).map(node_to_idx)
    if u.isna().any() or v.isna().any():
        raise ValueError("Edge list references a node missing from node_to_idx.")
    edges = list(zip(u.astype(int), v.astype(int)))
    return ig.Graph(n=n_nodes, edges=edges, directed=False)


def global_efficiency(g: ig.Graph, chunk_size: int = 500) -> float:
    """
    Global efficiency (Latora & Marchiori 2001): the mean of 1/d(i, j) over all
    ordered pairs i != j, where unreachable pairs contribute 0.

    Uses igraph's C-level unweighted BFS (Graph.distances) in row chunks so
    the full N x N distance matrix never has to be held in memory at once,
    which keeps this usable on the largest baseline networks in this project.

    :param g: Undirected graph (isolated nodes should already be included as vertices)
    :param chunk_size: Number of source nodes to compute shortest paths from per batch
    """
    n = g.vcount()
    if n < 2:
        return 0.0

    total = 0.0
    with np.errstate(divide='ignore'):
        for start in range(0, n, chunk_size):
            chunk = list(range(start, min(start + chunk_size, n)))
            dist = np.asarray(g.distances(source=chunk, target=None), dtype=float)
            total += np.where(dist > 0, 1.0 / dist, 0.0).sum()

    return total / (n * (n - 1))


def global_efficiency_from_edges(
    edge_df: pd.DataFrame, node_to_idx: dict, n_nodes: int, chunk_size: int = 500
) -> float:
    """Convenience wrapper: build the graph from an edge dataframe, then compute its global efficiency."""
    g = build_graph(edge_df, node_to_idx, n_nodes)
    return global_efficiency(g, chunk_size=chunk_size)
