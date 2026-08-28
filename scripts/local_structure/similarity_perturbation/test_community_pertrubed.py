import sys
import argparse
import numpy as np
import pandas as pd

from NoiseEffect.CommunityDetection.compare_perturbed_to_baseline import evaluate_network_repeats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path", type=str)
    parser.add_argument("baseline_npz", type=str)
    parser.add_argument("out_csv", type=str)
    parser.add_argument("algorithm", type=str)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    # 1. Load the pre-calculated baseline matrix and the strict node ordering
    npz_data = np.load(args.baseline_npz, allow_pickle=True)
    baseline_labels = npz_data['labels']
    node_order = npz_data['node_order']

    # Rebuild the exact dictionary mapping from the NPZ file
    N = len(node_order)
    node_to_idx = {str(node): i for i, node in enumerate(node_order)}

    # 2. Load and safely map the perturbed parquet file
    df_pert = pd.read_parquet(args.parquet_path)
    df_pert['source'] = df_pert['source'].astype(str).map(node_to_idx)
    df_pert['target'] = df_pert['target'].astype(str).map(node_to_idx)
    
    # Drop nodes destroyed by noise, cast to safe integers for igraph
    df_pert = df_pert.dropna(subset=['source', 'target']).astype({'source': int, 'target': int})

    # 3. Hand off to your core package
    seeds = [73942, 18405, 92051, 46138, 55920, 23084, 81763, 34591, 60247, 98316] # Define your seeds
    results = evaluate_network_repeats(
        df_pert=df_pert,
        n_nodes=N,
        algo=args.algorithm,
        seeds=seeds,
        baseline_labels=baseline_labels,
        n_jobs=args.n_jobs
    )

    # 4. Save results
    out_df = pd.DataFrame(results)
    out_df['network_id'] = args.parquet_path.split('/')[-1].replace('.parquet', '')
    out_df['algorithm'] = args.algorithm
    out_df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()