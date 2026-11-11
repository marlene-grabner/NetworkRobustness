"""
The global properties of one network. All functions take the igraph graph on
the full baseline node set (build_graph), so nodes that lost all their edges
are still counted.
"""

import igraph as ig
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

TOP_FRACTIONS = (0.01, 0.1)  # top nodes compared in centrality_rank
EFFICIENCY_SOURCES = 1000  # source nodes sampled for the global efficiency


def build_graph(edges: np.ndarray, n_nodes: int) -> ig.Graph:
    """Simple undirected graph on n_nodes nodes from an (m, 2) integer array."""
    g = ig.Graph(n=n_nodes, edges=edges.tolist(), directed=False)
    return g.simplify()


def gcc_and_singletons(g: ig.Graph) -> dict:
    """Fraction of the nodes in the largest component, and nodes without edges."""
    return {
        "gcc": max(g.connected_components().sizes()) / g.vcount(),
        "num_singletons": g.degree().count(0),
    }


def global_efficiency(
    g: ig.Graph, n_sources: int | None = EFFICIENCY_SOURCES, seed: int = 0
) -> dict:
    """
    Global efficiency (Latora & Marchiori 2001): the mean of 1/d(i, j) over
    all pairs i != j, unreachable pairs count 0.

    Approximation: inside the largest component the shortest paths are only
    computed from n_sources sampled source nodes; all smaller components are
    computed exactly. (Sampling over the whole network instead is much noisier
    once it fragments, because the result then depends on how many sources
    fall into the largest component.) Exact if n_sources is None or the
    largest component has no more nodes than that.
    """
    n = g.vcount()
    components = g.connected_components()
    largest = int(np.argmax(components.sizes()))
    membership = np.array(components.membership)

    gcc = g.induced_subgraph(np.flatnonzero(membership == largest).tolist())
    sources = np.arange(gcc.vcount())
    if n_sources is not None and n_sources < gcc.vcount():
        sources = np.random.default_rng(seed).choice(sources, n_sources, replace=False)
    total = _sum_inverse_distances(gcc, sources) * gcc.vcount() / len(sources)

    rest = g.induced_subgraph(
        np.flatnonzero((membership != largest) & (np.array(g.degree()) > 0)).tolist()
    )
    total += _sum_inverse_distances(rest, np.arange(rest.vcount()))
    return {"efficiency": float(total / (n * (n - 1)))}


def _sum_inverse_distances(g: ig.Graph, sources: np.ndarray) -> float:
    """Sum of 1/d from the source nodes to all other nodes, in chunks of sources."""
    total = 0.0
    with np.errstate(divide="ignore"):
        for start in range(0, len(sources), 500):
            chunk = sources[start : start + 500].tolist()
            dist = np.asarray(g.distances(source=chunk), dtype=float)
            total += np.where(dist > 0, 1.0 / dist, 0.0).sum()
    return total


def transitivity(g: ig.Graph) -> dict:
    """Global clustering coefficient: 3 x triangles / connected triples."""
    return {"transitivity": g.transitivity_undirected()}


def assortativity(g: ig.Graph) -> dict:
    """Degree assortativity: Pearson correlation of the degrees at edge ends."""
    return {"assortativity": g.assortativity_degree(directed=False)}


def kcore(g: ig.Graph) -> dict:
    """Highest k with a non-empty k-core, and the fraction of nodes in it."""
    core = np.array(g.coreness())
    return {
        "max_core": int(core.max()),
        "max_core_frac": float(np.mean(core == core.max())),
    }


def largest_eigenvalue(g: ig.Graph) -> dict:
    """Largest eigenvalue of the adjacency matrix (sparse Lanczos solver)."""
    n = g.vcount()
    edges = np.array(g.get_edgelist()).reshape(-1, 2)
    A = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = (A + A.T).tocsr()
    return {
        "largest_eigenvalue": float(
            eigsh(A, k=1, which="LA", return_eigenvectors=False)[0]
        )
    }


def centralities(g: ig.Graph) -> dict:
    """Degree and PageRank of every node; the reference for centrality_rank."""
    # PageRank times n is 1 on average. Rounded, so that nodes in equivalent
    # positions (e.g. all nodes without edges) are ties and not ranked by
    # floating point noise.
    pagerank = np.round(np.array(g.pagerank()) * g.vcount(), 8)
    return {"degree": np.array(g.degree()), "pagerank": pagerank}


def centrality_rank(g: ig.Graph, reference: dict) -> dict:
    """
    How well the node centralities (degree, PageRank) of the unperturbed
    network are kept (reference = centralities(baseline graph)):

      {name}_rank_corr        Spearman rank correlation over all nodes
      {name}_top{p}_overlap   fraction of the top p% nodes of the unperturbed
                              network that are still in the top p%
    """
    own = centralities(g)
    result = {}
    for name in own:
        result[f"{name}_rank_corr"] = float(
            spearmanr(own[name], reference[name]).statistic
        )
        for fraction in TOP_FRACTIONS:
            k = max(1, round(fraction * g.vcount()))
            overlap = np.intersect1d(_top(own[name], k), _top(reference[name], k))
            result[f"{name}_top{round(fraction * 100)}_overlap"] = len(overlap) / k
    return result


def _top(values: np.ndarray, k: int) -> np.ndarray:
    """The k nodes with the highest values; ties are broken by node index."""
    return np.argsort(-values, kind="stable")[:k]
