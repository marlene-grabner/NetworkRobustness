import networkx as nx
import igraph as ig


def erdos_renyi_model_from_graph(G: nx.Graph) -> nx.Graph:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    # Extract original nodes to retain node IDs
    original_nodes = list(G.nodes())
    mapping = dict(enumerate(original_nodes))

    # Generate G(n, m) ER graph
    G_er = nx.gnm_random_graph(n, m, seed=0)

    # Restore original node IDs
    G_er_labeled = nx.relabel_nodes(G_er, mapping)

    return G_er_labeled


def configuration_model_from_graph(G: nx.Graph) -> nx.Graph:
    original_nodes = list(G.nodes())
    degree_sequence = [G.degree(n) for n in original_nodes]
    mapping = dict(enumerate(original_nodes))

    # Generates a strictly simple, connected graph
    # Note: The input degree sequence must be connected/graphical
    g_ig = ig.Graph.Degree_Sequence(degree_sequence, method="vl")

    # Convert igraph back to NetworkX
    G_nx = g_ig.to_networkx()

    # Restore original node IDs
    G_cm = nx.relabel_nodes(G_nx, mapping)

    return G_cm


def sbm_model_from_graph():
    pass
