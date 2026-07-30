import sys
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd

from sklearn.metrics import roc_auc_score


# Add src folder to path
#sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import NoiseEffect.SeedExpansion.Expansion.io_helper as io
from NoiseEffect.SeedExpansion.Expansion import run_algorithm

# TODO: Adjust this import to point to your old DIAMOnD code location
from NoiseEffect.ModuleRecovery.ModuleDetectionAlgorithms import diamond

def main():
    # 1. Setup a small network for a fast verification

    def compute_macro_similarity(old_added_list, new_scores, index):
        # 1. Create a binary ground truth mask from the OLD implementation
        # 1 if the node was ever captured by the old code, 0 otherwise
        old_selected_set = set(str(node) for node, _ in old_added_list)
        
        y_true = np.zeros(index.n_nodes, dtype=int)
        for i, node_id in enumerate(index.nodes):
            if str(node_id) in old_selected_set:
                y_true[i] = 1
                
        # 2. Clean up new scores for sklearn (convert -inf to a safe low value)
        y_score = new_scores.copy()
        y_score[~np.isfinite(y_score)] = -1.0
        
        # 3. Compute Macro AUROC
        macro_auroc = roc_auc_score(y_true, y_score)
        print("\n=== Macro Similarity Analysis ===")
        print(f"Rank-Based Macro AUROC: {macro_auroc:.4f}")
        
        # 4. Compute Set Overlap at expanding milestones
        order = np.argsort(-new_scores, kind="stable")
        milestones = [5, 10, 20] # Adjust based on your X_NODES
        
        print(f"\n{'Window Size':<12} | {'Old Set Size':<12} | {'New Set Size':<12} | {'Jaccard Overlap':<15}")
        print("-" * 60)
        for k in milestones:
            if k > len(old_added_list):
                continue
            old_k_set = set(str(node) for node, _ in old_added_list[:k])
            new_k_set = set(str(index.nodes[idx]) for idx in order[:k])
            
            intersection = len(old_k_set & new_k_set)
            union = len(old_k_set | new_k_set)
            jaccard = intersection / union if union else 0.0
            
            print(f"{k:<12} | {len(old_k_set):<12} | {len(new_k_set):<12} | {jaccard:<15.4f}")


    network = "power"  
    edgelist_path = Path(f"data/baseline_networks/{network}.csv")
    seeds_csv = Path("./outputs/seed_expansion/synthetic_seeds/synthetic_seeds_by_bsf.csv")
    
    if not edgelist_path.exists():
        print(f"File not found: {edgelist_path}. Please check the path.")
        return

    print("=== Step 1: Building Graph Structures ===")
    edge_df = io.load_baseline_edgelist(edgelist_path)
    
    # New Pipeline format
    index = io.build_node_index(network, edge_df)
    adj_sparse, _ = io.edges_to_sparse(edge_df, index)
    
    # Old Pipeline format
    G_old = nx.from_pandas_edgelist(edge_df, source="source", target="target")
    
    # 2. Extract a sample seed configuration
    seed_table = io.load_seed_table(seeds_csv)
    network_seed_rows = seed_table[seed_table["network_id"] == network]
    if network_seed_rows.empty:
        print(f"No seeds found for {network}.")
        return
        
    sample_row = network_seed_rows.iloc[0]
    seed_id = sample_row["seed_id"]
    seed_nodes_list = str(sample_row["seed_nodes"]).split(";")
    
    # Align types for NetworkX lookups
    nx_node_type = type(next(iter(G_old.nodes())))
    seed_nodes_list = [nx_node_type(s) for s in seed_nodes_list]
    
    print(f"Testing Seed ID: {seed_id}")
    print(f"Seed Nodes count: {len(seed_nodes_list)}")

    # Set parameters: X=20 makes the check fast and clean to print
    X_NODES = 100
    ALPHA = 1

    print("\n=== Step 2: Running Old DIAMOnD Implementation ===")
    old_result = diamond(G_old, seed_nodes_list, X=X_NODES, alpha=ALPHA)
    # old_result.nodes_diamond is a list of tuples: [(node_name, p_value), ...]
    old_added_list = old_result.nodes_diamond

    print("\n=== Step 3: Running New Vectorized DIAMOnD Implementation ===")
    try:
        cast_seeds = np.array(seed_nodes_list, dtype=index.nodes.dtype)
    except (ValueError, TypeError):
        cast_seeds = np.array(seed_nodes_list)
    seed_idx = index.to_idx(cast_seeds)
    seed_idx = seed_idx[seed_idx >= 0]

    new_scores = run_algorithm(
        "diamond", 
        adj_sparse, 
        seed_idx, 
        params=dict(max_added_nodes=X_NODES, alpha=ALPHA)
    )

    # 4. Extract and sort the new scores
    # Higher score means added earlier. Nodes not added stay at -inf
    order = np.argsort(-new_scores, kind="stable")
    
    print("\n=== Step 4: Comparing Agglomeration Order ===")
    print(f"{'Rank':<5} | {'Old Node':<10} | {'Old P-Value':<13} | {'New Node':<10} | {'New Score':<10} | {'Match?':<6}")
    print("-" * 70)
    
    mismatches = 0
    # We only loop through the number of nodes actually returned by the old run
    for r in range(len(old_added_list)):
        old_node, old_p = old_added_list[r]
        
        # Pull the corresponding top node from the new vectorized output array
        new_node = index.nodes[order[r]]
        new_matrix_idx = order[r]
        new_score = new_scores[new_matrix_idx]
        
        match = "YES" if str(old_node) == str(new_node) else "NO"
        if str(old_node) != str(new_node):
            mismatches += 1
            
        print(f"{r+1:<5} | {str(old_node):<10} | {old_p:<13.4e} | {str(new_node):<10} | {new_score:<10.1f} | {match:<6}")

    x = compute_macro_similarity(old_added_list, new_scores, index)

    print(x)
    if mismatches == 0:
        print("\n SANITY CHECK PASSED: Both implementations chose the exact same nodes in the exact same sequence!")
    else:
        print(f"\n WARNING: Selection divergence detected. Found {mismatches} sequencing discrepancies.")

if __name__ == "__main__":
    main()


