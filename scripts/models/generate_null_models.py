# %%
from NoiseEffect import NullModelGeneration
from NoiseEffect import TopologicalProperties
import networkx as nx
import matplotlib.pyplot as plt

##################################################
# Load empirical networks
##################################################

networks_path = "./data/baseline_networks/"

G_ppi = nx.read_edgelist(networks_path + "chloe_ppi_lcc_2026_02_23.tsv", delimiter="\t")
G_power = nx.read_edgelist(networks_path + "western_us_power_grid.tsv", delimiter="\t")
G_collab = nx.read_edgelist(networks_path + "ca-AstroPh_gcc.tsv", delimiter="\t")
G_wiki = nx.read_edgelist(networks_path + "wiki-Vote_gcc.tsv", delimiter="\t")

# ====================================================================
# Null Models
# ====================================================================

##################################################
# Generate Erdos Renyi null model
##################################################

G_er_ppi = NullModelGeneration.erdos_renyi_model_from_graph(G_ppi)
G_er_power = NullModelGeneration.erdos_renyi_model_from_graph(G_power)
G_er_collab = NullModelGeneration.erdos_renyi_model_from_graph(G_collab)
G_er_wiki = NullModelGeneration.erdos_renyi_model_from_graph(G_wiki)

##################################################
# Generate degree-preserving null model
##################################################

G_cm_ppi = NullModelGeneration.configuration_model_from_graph(G_ppi)
G_cm_power = NullModelGeneration.configuration_model_from_graph(G_power)
G_cm_collab = NullModelGeneration.configuration_model_from_graph(G_collab)
G_cm_wiki = NullModelGeneration.configuration_model_from_graph(G_wiki)


# ====================================================================
# Evaluation
# ====================================================================

configuration_models = {
    "PPI": [G_ppi, G_er_ppi, G_cm_ppi],
    "Power Grid": [G_power, G_er_power, G_cm_power],
    "Collaboration": [G_collab, G_er_collab, G_cm_collab],
    "Wikipedia Vote": [G_wiki, G_er_wiki, G_cm_wiki],
}

for name, (G_orig, G_er, G_cm) in configuration_models.items():
    print("\n" + "=" * 50)
    print(f"Analysis of {name}:")
    print()
    # Comapare num nodes and edges
    print(
        f"Original network: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges"
    )
    print(
        f"Erdos-Renyi null model network: {G_er.number_of_nodes()} nodes, {G_er.number_of_edges()} edges"
    )
    print(
        f"Configuration model network: {G_cm.number_of_nodes()} nodes, {G_cm.number_of_edges()} edges"
    )

    # Compare properties of example node
    example_node = list(G_orig.nodes())[0]
    print(f"Node {example_node} degree in original: {G_orig.degree(example_node)}")
    print(f"Node {example_node} degree in ER model: {G_er.degree(example_node)}")
    print(
        f"Node {example_node} degree in configuration model: {G_cm.degree(example_node)}"
    )
    print(
        f"Intersection of neighbors for node {example_node} between original and ER model: {len(set(G_orig.neighbors(example_node)) & set(G_er.neighbors(example_node)))}"
    )
    print(
        f"Intersection of neighbors for node {example_node} between original and configuration model: {len(set(G_orig.neighbors(example_node)) & set(G_cm.neighbors(example_node)))}"
    )

    # Check fpor self-loops
    print(f"Number of self-loops in original network: {nx.number_of_selfloops(G_orig)}")
    print(f"Number of self-loops in ER model: {nx.number_of_selfloops(G_er)}")
    print(
        f"Number of self-loops in configuration model: {nx.number_of_selfloops(G_cm)}"
    )

    # Plot degree distributions
    TopologicalProperties.plot_degree_distribution(
        G_orig, title=f"{name} Original Network Degree Distribution"
    )
    TopologicalProperties.plot_degree_distribution(
        G_er, log_binning=False, title=f"{name} Erdos-Renyi Model Degree Distribution"
    )
    TopologicalProperties.plot_degree_distribution(
        G_cm, title=f"{name} Configuration Model Degree Distribution"
    )
    plt.show()

# ====================================================================
# Save the networks as .tsv files
# ====================================================================

# ER models
nx.write_edgelist(
    G_er_ppi,
    networks_path + "null_models/chloe_ppi_erdos_renyi.tsv",
    data=False,
    delimiter="\t",
)
nx.write_edgelist(
    G_er_power,
    networks_path + "null_models/western_us_power_grid_erdos_renyi.tsv",
    data=False,
    delimiter="\t",
)
nx.write_edgelist(
    G_er_collab,
    networks_path + "null_models/ca-AstroPh_erdos_renyi.tsv",
    data=False,
    delimiter="\t",
)
nx.write_edgelist(
    G_er_wiki,
    networks_path + "null_models/wiki-Vote_erdos_renyi.tsv",
    data=False,
    delimiter="\t",
)

# Configuration models
nx.write_edgelist(
    G_cm_ppi,
    networks_path + "null_models/chloe_ppi_configuration_model.tsv",
    data=False,
    delimiter="\t",
)
nx.write_edgelist(
    G_cm_power,
    networks_path + "null_models/western_us_power_grid_configuration_model.tsv",
    data=False,
    delimiter="\t",
)
nx.write_edgelist(
    G_cm_collab,
    networks_path + "null_models/ca-AstroPh_configuration_model.tsv",
    data=False,
    delimiter="\t",
)
nx.write_edgelist(
    G_cm_wiki,
    networks_path + "null_models/wiki-Vote_configuration_model.tsv",
    data=False,
    delimiter="\t",
)
