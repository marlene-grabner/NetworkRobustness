from fetch_extended_set import processNetwork

##################################################
# Fetches HuRI (Luck et al. 2020) as the main PPI of the core
# baseline set and saves it as data/baseline_networks/huri.csv.
# Same conversion as the extended set (undirected, simple, GCC,
# headerless csv). Nodes keep the Netzschleuder integer index; the mapping
# to Ensembl gene IDs is in nodes.csv inside the raw download
# (data/baseline_networks/non_gcc_ones_original_download/huri.csv.zip).
#
# The lab internal ppi.csv is NOT touched.
#
# Run from the repository root:
#   python scripts/baselines/fetch_core_huri.py
##################################################

HURI = {
    "key": "huri",
    "domain": "biological",
    "subdomain": "protein_interactions",
    "source": "netzschleuder",
    "dataset": "huri",
    "net": "HuRI",
    "directed": False,
}

row = processNetwork(
    HURI,
    output_folder="./data/baseline_networks",
    raw_folder="./data/baseline_networks/non_gcc_ones_original_download",
)
print({k: row[k] for k in ["status", "nodes_simple", "edges_simple", "self_loops_removed", "nodes_gcc", "edges_gcc", "gcc_fraction"]})
