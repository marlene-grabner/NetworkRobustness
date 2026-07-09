import networkx as nx
import pandas as pd

baseline_files = {
        "ppi": "data/baseline_networks/ppi.csv",
        "astro": "data/baseline_networks/astro.csv",
        "power":"data/baseline_networks/power.csv",
        "wiki": "data/baseline_networks/wiki.csv",
        "ppi_er": "data/baseline_networks/null_models/ppi_er.csv",
        "ppi_conf": "data/baseline_networks/null_models/ppi_conf.csv",
        "ppi_sbm": "data/baseline_networks/null_models/ppi_sbm.csv",
        "astro_er": "data/baseline_networks/null_models/astro_er.csv",
        "astro_conf": "data/baseline_networks/null_models/astro_conf.csv",
        "astro_sbm": "data/baseline_networks/null_models/astro_sbm.csv",
        "power_er": "data/baseline_networks/null_models/power_er.csv",
        "power_conf": "data/baseline_networks/null_models/power_conf.csv",
        "power_sbm": "data/baseline_networks/null_models/power_sbm.csv",
        "wiki_er": "data/baseline_networks/null_models/wiki_er.csv",
        "wiki_conf": "data/baseline_networks/null_models/wiki_conf.csv",
        "wiki_sbm": "data/baseline_networks/null_models/wiki_sbm.csv"
    }

results = []

for net_id, filepath in baseline_files.items():
    try:
        # Load the graph
        df = pd.read_csv(filepath, sep=',', header=None, names=['source','target'], dtype=str)
        G = nx.from_pandas_edgelist(df, 'source', 'target')
        
        # Calculate properties
        n_nodes = G.number_of_nodes()
        components = list(nx.connected_components(G))
        n_components = len(components)
        
        if n_components == 1:
            gcc_size = n_nodes
            gcc_percent = 100.0
        else:
            gcc_size = len(max(components, key=len))
            gcc_percent = (gcc_size / n_nodes) * 100
            
        results.append({
            "Network": net_id,
            "Nodes": n_nodes,
            "Components": n_components,
            "GCC Size": gcc_size,
            "GCC %": round(gcc_percent, 2)
        })
    except Exception as e:
        print(f"Error on {net_id}: {e}")

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))