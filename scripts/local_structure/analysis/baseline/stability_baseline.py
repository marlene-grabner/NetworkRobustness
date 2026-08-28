import numpy as np
import pandas as pd
import os
import glob
from itertools import combinations
from sklearn.metrics import adjusted_rand_score


################################################
# --- Functions ---
################################################

def analyze_community_npz(file_path):
    """
    Loads a .npz file containing community labels and calculates stability (ARI) 
    and structural metrics (sizes, singletons) across multiple runs.
    """
    # 1. Load the .npz file
    data = np.load(file_path)

    labels = data['labels']  # shape: (n_runs, n_nodes); degree-0 nodes are -1
    n_runs, n_nodes = labels.shape

    # Initialize lists to store metrics for each run/pair
    aris = []
    num_communities = []
    avg_comm_sizes = []
    num_singletons = []

    # 2. Calculate Pairwise ARI (Stability)
    # Degree-0 nodes are labeled -1 in both runs (same baseline network on both
    # sides), so masking on -1 excludes them - they're trivially "stable"
    # singletons in every run and would otherwise inflate the agreement score.
    # Get all unique pairs of runs (e.g., 20 runs = 190 pairs)
    for i, j in combinations(range(n_runs), 2):
        valid = (labels[i] != -1) & (labels[j] != -1)
        ari = adjusted_rand_score(labels[i][valid], labels[j][valid]) if valid.any() else np.nan
        aris.append(ari)
        
    # 3. Calculate Community Structural Stats per run
    for i in range(n_runs):
        run_labels = labels[i]
        
        # Get unique communities and how many nodes are in each
        unique_comms, counts = np.unique(run_labels, return_counts=True)
        
        n_comms = len(unique_comms)
        num_communities.append(n_comms)
        
        # Average community size is simply total nodes / number of communities
        avg_comm_sizes.append(n_nodes / n_comms)
        
        # Count how many communities have exactly 1 node
        singletons = np.sum(counts == 1)
        num_singletons.append(singletons)
        
    # 4. Aggregate metrics (Mean, Median, Std)
    file_name = os.path.basename(file_path)
    clean_name = file_name.replace('.npz', '')
    
    # Explicitly check for the two-word algorithm first
    if clean_name.endswith('_label_propagation'):
        algorithm = 'label_propagation'
        network = clean_name.replace('_label_propagation', '')
    else:
        # For single-word algorithms (louvain, leiden, infomap)
        parts = clean_name.split('_')
        algorithm = parts[-1]
        network = "_".join(parts[:-1])
    
    result = {
        'network': network,
        'algorithm': algorithm,
        'file_name': file_name,
        
        'ari_mean': np.mean(aris),
        'ari_median': np.median(aris),
        'ari_std': np.std(aris),
        
        'num_comms_mean': np.mean(num_communities),
        'num_comms_median': np.median(num_communities),
        'num_comms_std': np.std(num_communities),
        
        'avg_size_mean': np.mean(avg_comm_sizes),
        'avg_size_median': np.median(avg_comm_sizes),
        'avg_size_std': np.std(avg_comm_sizes),
        
        'singletons_mean': np.mean(num_singletons),
        'singletons_median': np.median(num_singletons),
        'singletons_std': np.std(num_singletons)
    }
    
    return result

def process_all_baselines(directory_path, output_csv="baseline_community_stats.csv"):
    """
    Iterates through all .npz files in a directory, processes them, and saves to CSV.
    """
    search_pattern = os.path.join(directory_path, "*.npz")
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        print(f"No .npz files found in {directory_path}")
        return
        
    print(f"Found {len(file_paths)} files. Processing...")
    
    all_results = []
    for fp in file_paths:
        try:
            stats = analyze_community_npz(fp)
            all_results.append(stats)
            print(f"Processed: {os.path.basename(fp)}")
        except Exception as e:
            print(f"Failed to process {fp}: {e}")
            
    # Convert to DataFrame and export
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nSuccess! Results saved to {output_csv}")

################################################
# --- Execution ---
################################################
output_folder = "outputs/local_structure/overview_csvs"

process_all_baselines('outputs/local_structure/baselines/infomap/', f'{output_folder}/baseline_community_stats_infomap.csv')
process_all_baselines('outputs/local_structure/baselines/louvain/', f'{output_folder}/baseline_community_stats_louvain.csv')
process_all_baselines('outputs/local_structure/baselines/leiden/', f'{output_folder}/baseline_community_stats_leiden.csv')
process_all_baselines('outputs/local_structure/baselines/label_propagation/', f'{output_folder}/baseline_community_stats_label_propagation.csv')