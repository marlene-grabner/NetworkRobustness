import os
import pandas as pd
import networkx as nx
from NoiseEffect.SeedExpansion.SyntheticSeeds import get_prioritized_candidates_by_percentile, generate_single_bfs_seed

def run_multi_degree_seed_pipeline(
    base_networks_dir: str, 
    target_percentiles: list[float], 
    num_repeats: int = 10, 
    num_seeds: int = 20
):
    """
    Iterates through baseline networks and multiple target degree tiers to generate
    localized BFS seed sets. Dynamically consumes the candidate pool until a full
    quota of successful seed sets is achieved for every single tier.
    """
    all_seed_records = []
    
    # Find all CSV edgelist files in your specified baseline directory
    network_files = [f for f in os.listdir(base_networks_dir) if f.endswith('.csv')]
    
    for filename in network_files:
        network_id = filename.split('.')[0]
        print(f"\n--- Processing network: {network_id} ---")
        
        # Load baseline network
        file_path = os.path.join(base_networks_dir, filename)
        G = nx.read_edgelist(file_path, delimiter=",")
        
        for target_percentile in target_percentiles:
            print(f"Generating seeds for target percentile: {target_percentile*100}%")
            
            # 1. Get the entire pool of nodes ordered by closeness to this percentile
            target_degree, candidate_pool = get_prioritized_candidates_by_percentile(G, target_percentile)
            
            # Shuffling a small window of perfect matches can break up absolute node-ID determinism
            # but keeping the global list sorted ensures we don't drift away from the target degree tier
            successful_count = 0
            
            # 2. Iterate down the pool until your quota is filled
            for start_node in candidate_pool:
                try:
                    # Try to extract the neighborhood
                    seed_list = generate_single_bfs_seed(G, start_node, num_seeds)
                except ValueError:
                    # Silently ignore the node if its component is too small, and try the next one!
                    continue
                
                # If BFS succeeds, process and record it
                actual_degree = G.degree(start_node)
                seed_id = f"{network_id}_start_node_{start_node}_degree_{actual_degree}"
                serialized_seeds = ";".join(map(str, seed_list))
                
                all_seed_records.append({
                    "network_id": network_id,
                    "target_percentile": target_percentile,
                    "resulting_target_degree": int(target_degree),
                    "actual_degree": actual_degree,
                    "start_node": start_node,
                    "seed_id": seed_id,
                    "seed_nodes": serialized_seeds
                })
                
                successful_count += 1
                
                # Stop immediately once you have collected your 10 distinct repeats
                if successful_count == num_repeats:
                    break
            
            print(f"    Successfully generated {successful_count}/{num_repeats} seed sets.")
            
            # Safety check if an extremely fractured network literally runs out of valid components
            if successful_count < num_repeats:
                print(f"    ⚠️ WARNING: Could only find {successful_count} valid components for this tier.")
                
    return all_seed_records
                
                
if __name__ == "__main__":
    # Example execution configuration
    seeds_baseline = run_multi_degree_seed_pipeline(
        base_networks_dir="./data/baseline_networks/",
        target_percentiles=[0.1, 0.5, 0.9],  # Sweeps across low, medium, and high connectivity zones
        num_repeats=5,               # 5 distinct seed sets per degree per network
        num_seeds=20                 # 20 nodes per local neighborhood module
    )

    seeds_null_models = run_multi_degree_seed_pipeline(
        base_networks_dir="./data/baseline_networks/null_models/",
        target_percentiles=[0.1, 0.5, 0.9],
        num_repeats=5,
        num_seeds=20
    )

    # Combine
    complete_seeds = seeds_baseline + seeds_null_models

    # Compile and save
    df_master_seeds = pd.DataFrame(complete_seeds)
    output_csv_path = "./outputs/seed_expansion/synthetic_seeds/synthetic_seeds_by_bsf.csv"
    df_master_seeds.to_csv(output_csv_path, index=False)
    print(f"Saved all multi-degree seeds to {output_csv_path}")

