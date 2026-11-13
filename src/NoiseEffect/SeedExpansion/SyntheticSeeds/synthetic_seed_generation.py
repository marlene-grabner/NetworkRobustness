"""
Synthetic seed sets for seed expansion. A module (a group of nodes that
belong together) is split at random into the seeds given to the algorithm
(seed_fraction of the module) and the hidden nodes the algorithm should
recover. Two kinds of module:

  bfs        a connected module of `module_size` nodes grown by BFS around a
             start node. Start nodes are drawn at random among the nodes whose
             degree is closest to a target degree percentile (low / medium /
             high connectivity). A module is rejected if its component is too
             small or if it overlaps too much (Jaccard) with a module that was
             already accepted for the same network, so the seed sets of a
             network are distinct.
  community  a whole community of a partition of the network (e.g. Leiden),
             drawn at random among the communities with min_size to max_size
             nodes. Modularity communities are often far larger (thousands of
             nodes), so communities above max_size are first split again with
             Leiden on their own subgraph, recursively, until every piece is
             small enough (split_large_communities). Every piece lies inside
             one community of the original partition. Communities are
             disjoint, so these seed sets never overlap.

Everything works on node indices 0..n-1 and a CSR adjacency matrix
(SeedExpansion.Expansion.edges_to_adjacency).
"""

import igraph as ig
import leidenalg as la
import numpy as np
from scipy import sparse


def bfs_module(
    adj: sparse.csr_matrix, start: int, size: int, rng: np.random.Generator
) -> np.ndarray | None:
    """
    The first `size` nodes reached by a BFS from `start` (start included), or
    None if its component is smaller. The neighbours of a node are visited in
    random order, so a high-degree start node does not always get the same
    (lowest index) neighbours.
    """
    visited = {start}
    order = [start]
    frontier = [start]
    while frontier:
        next_frontier = []
        for node in frontier:
            neighbours = adj.indices[adj.indptr[node] : adj.indptr[node + 1]]
            for nb in rng.permutation(neighbours):
                if nb in visited:
                    continue
                visited.add(nb)
                order.append(int(nb))
                if len(order) == size:
                    return np.array(order)
                next_frontier.append(nb)
        frontier = next_frontier
    return None


def candidates_by_degree(
    degrees: np.ndarray, target_percentile: float, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
    """
    Target degree (percentile of the degrees of the non-isolated nodes) and all
    non-isolated nodes ordered by the distance of their degree to it; nodes at
    the same distance are in random order.
    """
    if not 0.0 <= target_percentile <= 1.0:
        raise ValueError("target_percentile must be between 0 and 1")
    nodes = np.flatnonzero(degrees > 0)
    target_degree = float(np.percentile(degrees[nodes], target_percentile * 100))
    distance = np.abs(degrees[nodes] - target_degree)
    order = np.lexsort((rng.random(len(nodes)), distance))
    return target_degree, nodes[order]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / len(a | b)


def split_module(
    module: np.ndarray, seed_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Random split into round(seed_fraction * size) seeds and the hidden rest (both >= 1 node)."""
    n_seeds = min(max(round(seed_fraction * len(module)), 1), len(module) - 1)
    shuffled = rng.permutation(module)
    return np.sort(shuffled[:n_seeds]), np.sort(shuffled[n_seeds:])


def bfs_seed_sets(
    adj: sparse.csr_matrix,
    target_percentiles: list[float],
    n_sets: int,
    module_size: int,
    seed_fraction: float,
    max_overlap: float,
    rng: np.random.Generator,
    max_tries: int = 5000,
) -> list[dict]:
    """
    `n_sets` BFS modules per target percentile. Every module has a Jaccard
    overlap of at most `max_overlap` with every other module of this network
    (all percentiles). Returns one dict per module with node indices:
    target_percentile, target_degree, start_node, start_degree, seeds, hidden,
    max_overlap (largest Jaccard to an earlier module).

    A percentile can end with fewer than n_sets modules (small components or
    too much overlap within max_tries start nodes); a warning is printed.
    """
    if module_size < 2:
        raise ValueError("module_size must be at least 2")
    degrees = np.diff(adj.indptr)
    accepted, records = [], []
    for percentile in target_percentiles:
        target_degree, candidates = candidates_by_degree(degrees, percentile, rng)
        found = 0
        for start in candidates[:max_tries]:
            module = bfs_module(adj, int(start), module_size, rng)
            if module is None:
                continue
            overlaps = [jaccard(module, other) for other in accepted]
            if overlaps and max(overlaps) > max_overlap:
                continue
            accepted.append(module)
            seeds, hidden = split_module(module, seed_fraction, rng)
            records.append(
                {
                    "target_percentile": percentile,
                    "target_degree": target_degree,
                    "start_node": int(start),
                    "start_degree": int(degrees[start]),
                    "seeds": seeds,
                    "hidden": hidden,
                    "max_overlap": max(overlaps, default=0.0),
                }
            )
            found += 1
            if found == n_sets:
                break
        if found < n_sets:
            print(
                f"warning: only {found} of {n_sets} seed sets for percentile "
                f"{percentile} (target degree {target_degree:g})"
            )
    return records


def split_large_communities(
    adj: sparse.csr_matrix,
    labels: np.ndarray,
    max_size: int,
    seed: int,
    n_iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits every community above max_size again with Leiden (modularity, the
    settings of the community pipeline) on its own subgraph, recursively,
    until all pieces have at most max_size nodes. A community Leiden returns
    as a single piece can not be split and stays as it is (too large).

    :param labels: community label per node, -1 for nodes without a community
    :return: new labels (0..n_pieces-1, -1 as before), and per piece the label
        of the original community it lies in and the number of splits (0 =
        original community)
    """
    g = ig.Graph(
        n=adj.shape[0], edges=np.column_stack(sparse.triu(adj, k=1).nonzero()).tolist()
    )
    pieces = []  # (nodes, original community, depth)

    def split(nodes, parent, depth):
        if len(nodes) <= max_size:
            pieces.append((nodes, parent, depth))
            return
        partition = la.find_partition(
            g.subgraph(nodes.tolist()), la.ModularityVertexPartition,
            seed=seed, n_iterations=n_iterations,
        )
        if len(partition) == 1:
            pieces.append((nodes, parent, depth))
            return
        for community in partition:
            # subgraph keeps the order of nodes, so vertex i is nodes[i]
            split(nodes[np.array(community)], parent, depth + 1)

    for community in range(labels.max() + 1):
        nodes = np.flatnonzero(labels == community)
        if len(nodes):
            split(nodes, community, 0)

    new_labels = np.full_like(labels, -1)
    for i, (nodes, _, _) in enumerate(pieces):
        new_labels[nodes] = i
    parents = np.array([p for _, p, _ in pieces])
    depths = np.array([d for _, _, d in pieces])
    return new_labels, parents, depths


def community_seed_sets(
    labels: np.ndarray,
    n_sets: int,
    min_size: int,
    max_size: int,
    seed_fraction: float,
    rng: np.random.Generator,
    parents: np.ndarray | None = None,
    depths: np.ndarray | None = None,
) -> list[dict]:
    """
    `n_sets` communities drawn at random among those with min_size to
    max_size nodes (all of them if there are fewer; a warning is printed).

    :param labels: community label per node, -1 for nodes without a community
        (as CommunityDetection.detect_partitions returns them)
    :param parents, depths: from split_large_communities, stored with each set
    :return: one dict per community: community_id, community_size,
        parent_community_id, split_depth, seeds, hidden
    """
    sizes = np.bincount(labels[labels >= 0])
    eligible = np.flatnonzero((sizes >= min_size) & (sizes <= max_size))
    if len(eligible) < n_sets:
        print(
            f"warning: only {len(eligible)} of {n_sets} communities with "
            f"{min_size}-{max_size} nodes ({len(sizes)} communities)"
        )
    chosen = rng.choice(eligible, size=min(n_sets, len(eligible)), replace=False)
    records = []
    for community_id in np.sort(chosen):
        seeds, hidden = split_module(np.flatnonzero(labels == community_id), seed_fraction, rng)
        records.append(
            {
                "community_id": int(community_id),
                "community_size": int(sizes[community_id]),
                "parent_community_id": int(parents[community_id]) if parents is not None else int(community_id),
                "split_depth": int(depths[community_id]) if depths is not None else 0,
                "seeds": seeds,
                "hidden": hidden,
            }
        )
    return records
