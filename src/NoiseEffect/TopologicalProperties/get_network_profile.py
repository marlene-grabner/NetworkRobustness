import networkx as nx
import numpy as np
import scipy.stats as stats
import scipy.sparse.linalg as sla
import igraph as ig
import random


def get_network_profile(G):
    """
    Calculates a comprehensive topology profile for a NetworkX graph G.
    Uses SciPy and igraph for heavy computations to ensure extreme efficiency.
    """
    # Ensure graph is undirected and simple for baseline metrics
    G = nx.Graph(G)

    # 1. Global properties
    print("Calculating macroscopic properties...")
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)

    # Giant Connected Component (GCC) fraction
    gcc_nodes = max(nx.connected_components(G), key=len)
    gcc_fraction = len(gcc_nodes) / nodes

    # 2. Degree statistics
    print("Calculating degree statistics...")
    degrees = np.array([d for n, d in G.degree()])
    deg_skew = stats.skew(degrees)
    deg_cv = np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else 0

    # Assortativity
    assortativity = nx.degree_assortativity_coefficient(G)

    # 3. Spectral proeprties
    # Use the unnormalized Laplacian matrix
    print("Calculating spectral properties...")
    L = nx.laplacian_matrix(G).astype(float)

    # We look for the 3 smallest algebraic eigenvalues (SA).
    # lambda_1 is always ~0. lambda_2 is algebraic connectivity.
    try:
        # tol=1e-5
        eigenvalues, _ = sla.eigsh(L, k=3, which="SA", tol=1e-5)
        eigenvalues = np.sort(eigenvalues)
        alg_connectivity = eigenvalues[1]
        spectral_gap = eigenvalues[2] - eigenvalues[1]
    except Exception as e:
        # Fallback if eigsh fails to converge
        alg_connectivity, spectral_gap = 0.0, 0.0

    # 4. Transitivity, clustering, and modularity
    print("Calculating transitivity, clustering, and modularity with igraph...")
    ig_g = ig.Graph.from_networkx(G)

    # Transitivity (Global clustering) and Average Local Clustering
    transitivity = ig_g.transitivity_undirected()
    clustering = ig_g.transitivity_avglocal_undirected()

    # Modularity using Louvain (Multilevel)
    partition = ig_g.community_multilevel()
    modularity = partition.modularity

    # 5. Path length
    print("Calculating average path lengths...")
    avg_path_length = ig_g.average_path_length(directed=False)

    # Compile results
    profile = {
        "Nodes": nodes,
        "Edges": edges,
        "Density": density,
        "GCC_Fraction": gcc_fraction,
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

    return profile
