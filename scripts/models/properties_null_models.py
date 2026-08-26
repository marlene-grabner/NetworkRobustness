import networkx as nx
import pandas as pd
from NoiseEffect import TopologicalProperties
from NoiseEffect.utils import df_to_latex, save_latex_tables


############################################################
# Load the baseline network
############################################################

networks_path = "./data/baseline_networks/null_models/"


G_ppi_er = nx.read_edgelist(networks_path + "ppi_er.csv", delimiter=",")
G_power_er = nx.read_edgelist(networks_path + "power_er.csv", delimiter=",")
G_collab_er = nx.read_edgelist(networks_path + "astro_er.csv", delimiter=",")
G_wiki_er = nx.read_edgelist(networks_path + "wiki_er.csv", delimiter=",")

G_ppi_conf = nx.read_edgelist(networks_path + "ppi_conf.csv", delimiter=",")
G_power_conf = nx.read_edgelist(networks_path + "power_conf.csv", delimiter=",")
G_collab_conf = nx.read_edgelist(networks_path + "astro_conf.csv", delimiter=",")
G_wiki_conf = nx.read_edgelist(networks_path + "wiki_conf.csv", delimiter=",")

G_ppi_sbm = nx.read_edgelist(networks_path + "ppi_sbm.csv", delimiter=",")
G_power_sbm = nx.read_edgelist(networks_path + "power_sbm.csv", delimiter=",")
G_collab_sbm = nx.read_edgelist(networks_path + "astro_sbm.csv", delimiter=",")
G_wiki_sbm = nx.read_edgelist(networks_path + "wiki_sbm.csv", delimiter=",")


prop_ppi_er = TopologicalProperties.get_network_profile(G_ppi_er)
prop_power_er = TopologicalProperties.get_network_profile(G_power_er)
prop_collab_er = TopologicalProperties.get_network_profile(G_collab_er)
prop_wiki_er = TopologicalProperties.get_network_profile(G_wiki_er)

prop_ppi_conf = TopologicalProperties.get_network_profile(G_ppi_conf)
prop_power_conf = TopologicalProperties.get_network_profile(G_power_conf)
prop_collab_conf = TopologicalProperties.get_network_profile(G_collab_conf)
prop_wiki_conf = TopologicalProperties.get_network_profile(G_wiki_conf)

prop_ppi_sbm = TopologicalProperties.get_network_profile(G_ppi_sbm)
prop_power_sbm = TopologicalProperties.get_network_profile(G_power_sbm)
prop_collab_sbm = TopologicalProperties.get_network_profile(G_collab_sbm)
prop_wiki_sbm = TopologicalProperties.get_network_profile(G_wiki_sbm)


df = pd.DataFrame.from_dict(
    [
        prop_ppi_er,
        prop_power_er,
        prop_collab_er,
        prop_wiki_er,
        prop_ppi_conf,
        prop_power_conf,
        prop_collab_conf,
        prop_wiki_conf,
        prop_ppi_sbm,
        prop_power_sbm,
        prop_collab_sbm,
        prop_wiki_sbm,
    ]
).T

df.rename(
    columns={
        0: "PPI (ER)",
        1: "Power Grid (ER)",
        2: "Astrophysics (ER)",
        3: "Wikipedia (ER)",
        4: "PPI (Conf)",
        5: "Power Grid (Conf)",
        6: "Astrophysics (Conf)",
        7: "Wikipedia (Conf)",
        8: "PPI (SBM)",
        9: "Power Grid (SBM)",
        10: "Astrophysics (SBM)",
        11: "Wikipedia (SBM)",
    },
    inplace=True,
)

df.to_csv("./outputs/models/null_model_network_properties.csv", index=True, header=True)

# Save the table as a LaTeX file
latex_table = df_to_latex(
    df=df,
    caption="Topological Properties of Null Models",
    label="tab:null_model_network_properties",
)

save_latex_tables(
    file_path="outputs/latex_tables/models/topological_properties/null_model_network_properties.tex",
    tables=latex_table,
)
