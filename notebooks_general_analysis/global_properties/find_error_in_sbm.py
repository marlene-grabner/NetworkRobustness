#%%
import pandas as pd

def validate_network_consistency(baseline_path: str, perturbed_paths: list[str]):
    """
    Validates node consistency between a baseline network and perturbed networks.
    """
    print(f"--- Validating Baseline: {baseline_path} ---")
    
    # FIX: Force string type on load so '1' matches '1'
    df_base = pd.read_csv(baseline_path, sep='\t', header=None, names=['source', 'target'], dtype=str)
    baseline_nodes = set(df_base['source']).union(set(df_base['target']))
    print(f"Baseline Nodes: {len(baseline_nodes)}")
    print(f"Baseline Edges: {len(df_base)}")
    
    # 2. Check Perturbed Files
    for path in perturbed_paths:
        print(f"\n--- Checking Perturbed File: {path.split('/')[-1]} ---")
        df_pert = pd.read_parquet(path)
        
        # We only need to check the first repeat (repeat == 0 or 1) for a quick sanity check
        first_repeat_id = df_pert['repeat'].iloc[0]
        df_rep = df_pert[df_pert['repeat'] == first_repeat_id].copy()
        
        # FIX: Force string type on the perturbed nodes before creating the set
        pert_nodes = set(df_rep['source'].astype(str)).union(set(df_rep['target'].astype(str)))
        
        print(f"Repeat ID: {first_repeat_id}")
        print(f"Perturbed Nodes (Active): {len(pert_nodes)}")
        print(f"Perturbed Edges: {len(df_rep)}")
        
        # --- THE CRITICAL CHECKS ---
        
        # Nodes in perturbed that DO NOT exist in baseline
        rogue_nodes = pert_nodes - baseline_nodes
        if len(rogue_nodes) > 0:
            print(f"❌ ERROR: Found {len(rogue_nodes)} rogue nodes in perturbed graph not present in baseline!")
            # Print a few examples of the rogue nodes
            print(f"   Examples of rogue nodes: {list(rogue_nodes)[:10]}")
        else:
            print("✅ All perturbed nodes exist in the baseline vocabulary.")
            
        # Nodes in baseline that DO NOT exist in perturbed (These are your singletons!)
        missing_nodes = baseline_nodes - pert_nodes
        print(f"Expected Singletons for this repeat: {len(missing_nodes)}")
        
        if len(pert_nodes) > len(baseline_nodes):
            print("❌ ERROR: Perturbed graph has a larger absolute node count than the baseline!")



if __name__ == '__main__':
    # Put your baseline path here
    b_path = "./data/baseline_networks/null_models/wiki-Vote_sbm.tsv"
    
    # Put 3 or 4 paths to the specific .parquet files that were giving negative singletons
    p_paths = [
        "./data/perturbed_networks/wiki-Vote_sbm/perturbed_hub_target/wiki-Vote_gcc_sbm_targeted_hub_addition_noise_0p2.parquet",
        "./data/perturbed_networks/wiki-Vote_sbm/perturbed_random_target/wiki-Vote_gcc_sbm_added_edges_noise_0p3.parquet"
    ]
    
    validate_network_consistency(b_path, p_paths)
