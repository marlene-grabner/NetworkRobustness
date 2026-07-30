import sys
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import NoiseEffect.SeedExpansion.Expansion.io_helper as io
from NoiseEffect.SeedExpansion.Expansion import run_algorithm
# TODO: Adjust this import path to where your old RWR implementation lives
from NoiseEffect.ModuleRecovery.ModuleDetectionAlgorithms import randomWalkWithRestartRowNormalization

def main():
    # 1. Pick a small network for a fast check (e.g., 'power' or a slice of 'astro')
    network = "power"  
    edgelist_path = Path(f"data/baseline_networks/{network}.csv")
    seeds_csv = Path("./outputs/seed_expansion/synthetic_seeds/synthetic_seeds_by_bsf.csv")
    
    if not edgelist_path.exists():
        print(f"File not found: {edgelist_path}. Please adjust the network name.")
        return

    print("=== Step 1: Loading Data and Building Graph Representations ===")
    edge_df = io.load_baseline_edgelist(edgelist_path)
    
    # New Pipeline Format: NodeIndex and Sparse CSR Matrix
    index = io.build_node_index(network, edge_df)
    adj_sparse, _ = io.edges_to_sparse(edge_df, index)
    
    # Old Pipeline Format: NetworkX Graph
    G_old = nx.from_pandas_edgelist(edge_df, source="source", target="target")
    
    # 2. Grab a test seed from your seed table
    seed_table = io.load_seed_table(seeds_csv)
    network_seed_rows = seed_table[seed_table["network_id"] == network]
    if network_seed_rows.empty:
        print(f"No seeds found for {network} in seed table.")
        return
        
    sample_row = network_seed_rows.iloc[0]
    seed_id = sample_row["seed_id"]
    seed_nodes_list = str(sample_row["seed_nodes"]).split(";")
    
    # Detect the node type used by NetworkX and cast the seeds to match it
    nx_node_type = type(next(iter(G_old.nodes())))
    seed_nodes_list = [nx_node_type(s) for s in seed_nodes_list]
    seed_nodes_set = set(seed_nodes_list)  # Converted to set for efficient, type-safe filtering
    
    print(f"Testing Seed ID: {seed_id}")
    print(f"Seed Nodes: {seed_nodes_list}")

    # Match parameters explicitly!
    RESTART = 0.85
    TOL = 1e-6
    MAX_ITER = 1000

    print("\n=== Step 2: Running Old NetworkX Implementation ===")
    old_result = randomWalkWithRestartRowNormalization(
        G=G_old,
        seed_nodes=seed_nodes_list,
        restart=RESTART,
        tol=TOL,
        max_iter=MAX_ITER
    )
    old_ranking_labels = list(old_result.nodes_ranked.keys())

    print("\n=== Step 3: Running New Vectorized Implementation ===")
    try:
        cast_seeds = np.array(seed_nodes_list, dtype=index.nodes.dtype)
    except (ValueError, TypeError):
        cast_seeds = np.array(seed_nodes_list)
    seed_idx = index.to_idx(cast_seeds)
    seed_idx = seed_idx[seed_idx >= 0]

    new_scores = run_algorithm(
        "rwr_row", 
        adj_sparse, 
        seed_idx, 
        params=dict(restart=RESTART, tol=TOL, max_iter=MAX_ITER)
    )

    # 4. Extract ranking labels from the new scores
    order = np.argsort(-new_scores, kind="stable")
    new_ranking_labels = index.nodes[order].tolist()
    
    # Fixed: Type-safe filter to drop seeds from the new ranking
    new_ranking_labels = [node for node in new_ranking_labels if node not in seed_nodes_set]

    print("\n=== Step 4: Comparing Rankings ===")
    top_k = min(250, len(old_ranking_labels))
    
    # Updated headers to fit probability data columns
    print(f"{'Rank':<5} | {'Old Label':<12} | {'Old Prob':<13} | {'New Label':<12} | {'New Prob':<13} | {'Match?':<6}")
    print("-" * 80)
    
    mismatches = 0
    for r in range(top_k):
        old_l = old_ranking_labels[r]
        new_l = new_ranking_labels[r]
        
        # Pull probabilities from old dictionary output
        old_prob = old_result.nodes_ranked[old_l]
        
        # Map new label back to its matrix index to get the raw vector score
        new_matrix_idx = index.node_to_idx[new_l]
        new_prob = new_scores[new_matrix_idx]
        
        match = "YES" if str(old_l) == str(new_l) else "NO"
        if old_l != new_l:
            mismatches += 1
            
        print(f"{r+1:<5} | {str(old_l):<12} | {old_prob:<13.6e} | {str(new_l):<12} | {new_prob:<13.6e} | {match:<6}")

    check_len = min(len(old_ranking_labels), len(new_ranking_labels))
    
    old_check = [str(x) for x in old_ranking_labels[:check_len]]
    new_check = [str(x) for x in new_ranking_labels[:check_len]]
    
    if old_check == new_check:
        print(f"\n SANITY CHECK PASSED: Both algorithms produce the exact same ranking sequence across all {check_len} comparable nodes!")
    else:
        print(f"\n WARNING: Rankings diverge. Found {mismatches} mismatches inside the top {top_k} evaluations.")

if __name__ == "__main__":
    main()