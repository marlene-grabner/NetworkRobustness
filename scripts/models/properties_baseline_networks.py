import networkx as nx
import pandas as pd
from NoiseEffect import TopologicalProperties
from NoiseEffect.utils import df_to_latex, save_latex_tables

############################################################
# Load the baseline network
############################################################

networks_path = "./data/baseline_networks/"


G_ppi = nx.read_edgelist(networks_path + "ppi.csv", delimiter=",")
G_power = nx.read_edgelist(networks_path + "power.csv", delimiter=",")
G_collab = nx.read_edgelist(networks_path + "astro.csv", delimiter=",")
G_wiki = nx.read_edgelist(networks_path + "wiki.csv", delimiter=",")


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


df.to_csv("./outputs/models/baseline_network_properties.csv", index=True, header=True)

# Save the table as a LaTeX file
latex_table = df_to_latex(
    df=df,
    caption="Topological Properties of Baseline Networks",
    label="tab:baseline_network_properties",
    column_format="lcccccccccccc",
)

save_latex_tables(
    file_path="outputs/latex_tables/models/topological_properties/baseline_network_properties.tex",
    tables=latex_table,
)


############################################################
# Plot degree distribution
############################################################

TopologicalProperties.plot_degree_distribution(
    G_ppi,
    num_bins=40,
    log_binning=True,
    fit_trend=True,
    save_fig="./outputs/models/figures/baseline_properties/degree_distributions/ppi_degree_distribution.pdf",
    color="#782235",
    marker="o",
)
TopologicalProperties.plot_degree_distribution(
    G_power,
    num_bins=40,
    log_binning=False,
    fit_trend=True,
    save_fig="./outputs/models/figures/baseline_properties/degree_distributions/power_grid_degree_distribution.pdf",
    color="#372278",
    marker="o",
)
TopologicalProperties.plot_degree_distribution(
    G_collab,
    num_bins=40,
    log_binning=True,
    fit_trend=True,
    save_fig="./outputs/models/figures/baseline_properties/degree_distributions/astrophysics_degree_distribution.pdf",
    color="#227851",
    marker="o",
)
TopologicalProperties.plot_degree_distribution(
    G_wiki,
    num_bins=40,
    save_fig="./outputs/models/figures/baseline_properties/degree_distributions/wiki_degree_distribution.pdf",
    log_binning=True,
    fit_trend=True,
    color="#E8AD0C",
    marker="o",
)
