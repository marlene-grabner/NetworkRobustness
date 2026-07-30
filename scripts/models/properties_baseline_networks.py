import networkx as nx
import pandas as pd
from NoiseEffect import TopologicalProperties


############################################################
# Load the baseline network
############################################################

networks_path = "./data/baseline_networks/"


G_ppi = nx.read_edgelist(networks_path + "chloe_ppi_lcc_2026_02_23.tsv", delimiter="\t")
G_power = nx.read_edgelist(networks_path + "western_us_power_grid.tsv", delimiter="\t")
G_collab = nx.read_edgelist(networks_path + "ca-AstroPh_gcc.tsv", delimiter="\t")
G_wiki = nx.read_edgelist(networks_path + "wiki-Vote_gcc.tsv", delimiter="\t")


prop_ppi = TopologicalProperties.get_network_profile(G_ppi)
prop_power = TopologicalProperties.get_network_profile(G_power)
prop_collab = TopologicalProperties.get_network_profile(G_collab)
prop_wiki = TopologicalProperties.get_network_profile(G_wiki)


df = pd.DataFrame.from_dict([prop_ppi, prop_power, prop_collab, prop_wiki]).T

df.rename(
    columns={
        0: "Protein-Protein Interaction Network",
        1: "Western US Power Grid Network",
        2: "Astrophysics Collaboration Network",
        3: "Wikipedia Vote Network",
    },
    inplace=True,
)


############################################################
# Plot degree distribution
############################################################

TopologicalProperties.plot_degree_distribution(
    G_ppi,
    num_bins=40,
    log_binning=True,
    fit_trend=True,
    save_fig="./outputs/figures/baseline_properties/ppi_degree_distribution.pdf",
    color="#782235",
    marker="o",
)
TopologicalProperties.plot_degree_distribution(
    G_power,
    num_bins=40,
    log_binning=False,
    fit_trend=True,
    save_fig="./outputs/figures/baseline_properties/power_grid_degree_distribution.pdf",
    color="#372278",
    marker="o",
)
TopologicalProperties.plot_degree_distribution(
    G_collab,
    num_bins=40,
    log_binning=True,
    fit_trend=True,
    save_fig="./outputs/figures/baseline_properties/astrophysics_degree_distribution.pdf",
    color="#227851",
    marker="o",
)
TopologicalProperties.plot_degree_distribution(
    G_wiki,
    num_bins=40,
    save_fig="./outputs/figures/baseline_properties/wiki_degree_distribution.pdf",
    log_binning=True,
    fit_trend=True,
    color="#E8AD0C",
    marker="o",
)
