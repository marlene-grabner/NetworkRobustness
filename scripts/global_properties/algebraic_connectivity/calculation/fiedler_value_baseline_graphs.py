import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from NoiseEffect.GlobalProperties import fiedler_on_gcc

def calculate_baselines():
    # Path to the 16 baseline networks csv files
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
        print(f"Processing baseline: {net_id}")
        
        try:
            # Note: Changed to sep='\t' assuming these are TSV files
            df_base = pd.read_csv(
                filepath, 
                sep=',', 
                header=None, 
                names=['source','target'], 
                dtype=str
            )
            
            # Build node index
            baseline_nodes = list(set(df_base['source']) | set(df_base['target']))
            N = len(baseline_nodes)
            node_to_idx = {n: i for i, n in enumerate(baseline_nodes)}
            
            # Map nodes to integer indices
            u = df_base['source'].map(node_to_idx).values
            v = df_base['target'].map(node_to_idx).values
            
            # Drop any NaNs just in case
            valid = ~np.isnan(u) & ~np.isnan(v)
            u, v = u[valid].astype(int), v[valid].astype(int)

            # Build the symmetric, unweighted sparse adjacency matrix
            A = coo_matrix((np.ones(len(u)), (u, v)), shape=(N, N))
            A = A.maximum(A.T)
            A.data = np.ones_like(A.data)

            # Calculate using your custom C function
            fiedler, was_disconnected = fiedler_on_gcc(A, N)

            results.append({
                'network_id': net_id,
                'algebraic_connectivity': fiedler,
                'was_disconnected': was_disconnected,
            })
            
        except Exception as e:
            print(f"Failed to process {net_id}: {e}")

    # Save outputs
    out_path = "baseline_algebraic_connectivity.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Done computing baselines! Saved to {out_path}")

if __name__ == '__main__':
    calculate_baselines()