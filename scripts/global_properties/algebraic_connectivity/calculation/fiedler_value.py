import os, sys
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from NoiseEffect.GlobalProperties import fiedler_on_gcc

def main():
    parquet_path   = sys.argv[1]   # e.g. /data/perturbed/hub/noise_05_rep42.parquet
    baseline_path  = sys.argv[2]   # e.g. /data/baseline/network_A.tsv
    out_path       = sys.argv[3]   # e.g. /outputs/hub/noise_05_rep42.csv
    
    # Build node index from baseline
    df_base = pd.read_csv(
            baseline_path, 
            sep=',', 
            header=None, 
            names=['source','target'], 
            dtype=str
        )
    
    baseline_nodes = list(set(df_base['source']) | set(df_base['target']))
    N = len(baseline_nodes)
    node_to_idx = {n: i for i, n in enumerate(baseline_nodes)}

    df_pert = pd.read_parquet(parquet_path)
    results = []

    for repeat_id, group in df_pert.groupby('repeat'):
        u = group['source'].map(node_to_idx).values
        v = group['target'].map(node_to_idx).values
        valid = ~np.isnan(u) & ~np.isnan(v)
        u, v = u[valid].astype(int), v[valid].astype(int)

        A = coo_matrix((np.ones(len(u)), (u, v)), shape=(N, N))
        A = A.maximum(A.T)
        A.data = np.ones_like(A.data)

        fiedler, was_disconnected = fiedler_on_gcc(A, N)

        results.append({
            'network_id': os.path.basename(parquet_path).replace('.parquet','') + f"_repeat_{repeat_id}",
            'algebraic_connectivity': fiedler,
            'was_disconnected': was_disconnected,
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Done: {out_path}")

if __name__ == '__main__':
    main()
