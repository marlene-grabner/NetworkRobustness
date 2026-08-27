import networkx as nx
import numpy as np
import scipy.stats as stats
import scipy.sparse.linalg as sla
import igraph as ig
import random


def get_network_profile(G, metrics_on_gcc=True, verbose=False):
    """
    Topology profile for a NetworkX graph G.

    Connectivity diagnostics (components, singletons, GCC fraction) are always
    computed on the FULL graph -- that's the point of inspecting null models.

    Structural + spectral metrics (spectral gap, algebraic connectivity, path
    length, clustering, modularity, degree stats) are computed on the GCC when
    metrics_on_gcc=True. This is required for the numbers to mean the same
    thing they mean for the (fully connected) empirical networks:
      - algebraic connectivity of a disconnected graph is exactly 0 by
        definition (multiplicity of the zero eigenvalue = number of
        components), so whole-graph spectra are uninformative for the nulls
      - igraph's average_path_length silently averages only reachable pairs
        on a disconnected graph, giving a non-comparable value
    Set metrics_on_gcc=False only if you specifically want whole-graph stats.
    """
    G = nx.Graph(G)  # undirected, simple

    def log(msg):
        if verbose:
            print(msg)

    # -------- Connectivity diagnostics (ALWAYS on the full graph) --------
    log("Connectivity diagnostics on full graph...")
    n_full = G.number_of_nodes()
    m_full = G.number_of_edges()
    components = list(nx.connected_components(G))
    n_components = len(components)
    gcc_nodes = max(components, key=len)
    gcc_size = len(gcc_nodes)
    gcc_fraction = gcc_size / n_full if n_full else 0.0
    # singletons = nodes with degree 0 (isolated)
    degrees_full = np.array([d for _, d in G.degree()])
    n_singletons = int(np.sum(degrees_full == 0))
    n_nongcc_nodes = n_full - gcc_size

    # -------- Choose the graph structural metrics run on --------
    if metrics_on_gcc and gcc_fraction < 1.0:
        log(f"Extracting GCC ({gcc_size}/{n_full} nodes) for structural metrics...")
        H = G.subgraph(gcc_nodes).copy()
        metrics_scope = "GCC"
    else:
        H = G
        metrics_scope = "full" if gcc_fraction == 1.0 else "GCC(=full)"

    n = H.number_of_nodes()
    m = H.number_of_edges()
    density = nx.density(H)

    # -------- Degree statistics (on H) --------
    log("Degree statistics...")
    degrees = np.array([d for _, d in H.degree()])
    deg_skew = float(stats.skew(degrees))
    deg_cv = float(np.std(degrees) / np.mean(degrees)) if degrees.mean() > 0 else 0.0
    assortativity = nx.degree_assortativity_coefficient(H)

    # -------- Spectral properties (on H, now guaranteed connected) --------
    log("Spectral properties...")
    L = nx.laplacian_matrix(H).astype(float)
    try:
        eigvals = sla.eigsh(L, k=6, sigma=1e-8, which="LM", return_eigenvectors=False)
        eigvals = np.sort(eigvals)
        alg_connectivity = float(eigvals[1])
        spectral_gap = float(eigvals[2] - eigvals[1])
    except Exception:
        alg_connectivity, spectral_gap = np.nan, np.nan  # NaN, not 0 -- see note

    # -------- Clustering, modularity, path length (igraph, on H) --------
    log("Clustering / modularity / path length...")
    ig_g = ig.Graph.from_networkx(H)
    transitivity = ig_g.transitivity_undirected()
    clustering = ig_g.transitivity_avglocal_undirected()
    modularity = ig_g.community_multilevel().modularity
    avg_path_length = ig_g.average_path_length(directed=False)

    return {
        # full-graph connectivity diagnostics
        "Nodes_Full": n_full,
        "Edges_Full": m_full,
        "N_Components": n_components,
        "N_Singletons": n_singletons,
        "N_NonGCC_Nodes": n_nongcc_nodes,
        "GCC_Size": gcc_size,
        "GCC_Fraction": gcc_fraction,
        "Metrics_Scope": metrics_scope,
        "Nodes": n,
        "Edges": m,
        "Density": density,
        "Degree_Skew": deg_skew,
        "Degree_CV": deg_cv,
        "Assortativity": assortativity,
        "Transitivity": transitivity,
        "Clustering_Coefficient": clustering,
        "Modularity_Louvain": modularity,
        "Algebraic_Connectivity": alg_connectivity,
        "Spectral_Gap": spectral_gap,
        "Avg_Path_Length": avg_path_length,
    }
