import networkx as nx
import pandas as pd
from NoiseEffect import TopologicalProperties
from NoiseEffect.utils import df_to_latex, save_latex_tables

##################################################
# Topological properties of the extended baseline set.
# Run from the repository root after fetch_extended_set.py:
#   python scripts/baselines/properties_extended_set.py
##################################################

networks_path = "./data/baseline_networks/extended_set/"
overview_file = networks_path + "extended_set_overview.csv"
output_csv = "./outputs/models/extended_set_network_properties.csv"
output_latex = "outputs/latex_tables/models/topological_properties/extended_set_network_properties.tex"

# Columns shown in the LaTeX table. All networks are stored as their GCC,
# so the full-graph connectivity diagnostics are left out of the table
# (they are still in the csv).
table_columns = [
    "Domain",
    "Subdomain",
    "Nodes",
    "Edges",
    "Average_Degree",
    "Density",
    "Degree_Skew",
    "Degree_CV",
    "Assortativity",
    "Transitivity",
    "Clustering_Coefficient",
    "Modularity_Louvain",
    "Algebraic_Connectivity",
    "Spectral_Gap",
    "Avg_Path_Length",
]

############################################################
# Calculate the properties
############################################################

overview = pd.read_csv(overview_file, dtype={"net": str})
overview = overview[overview["status"] == "ok"]

profiles = {}
for _, row in overview.iterrows():
    print(f"Profiling {row['key']}...")
    G = nx.read_edgelist(networks_path + f"{row['key']}.csv", delimiter=",")
    profile = TopologicalProperties.get_network_profile(G)
    profiles[row["key"]] = {
        "Domain": row["domain"],
        "Subdomain": row["subdomain"],
        "Directed_Original": row["directed"],
        **profile,
    }

# Networks as rows: with ~40 networks a column per network does not fit
df = pd.DataFrame.from_dict(profiles, orient="index")
df.index.name = "Network"

df.to_csv(output_csv, index=True, header=True)

############################################################
# Save the table as a LaTeX file
############################################################

df_table = df[table_columns].copy()
df_table["Subdomain"] = df_table["Subdomain"].str.replace("_", " ")

latex_table = df_to_latex(
    df=df_table,
    caption="Topological Properties of the Extended Set of Baseline Networks",
    label="tab:extended_set_network_properties",
    header_max_width=12,
)

save_latex_tables(file_path=output_latex, tables=latex_table)
