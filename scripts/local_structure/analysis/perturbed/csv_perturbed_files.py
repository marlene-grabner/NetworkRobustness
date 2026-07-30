import os
import glob
import pandas as pd
import numpy as np

def aggregate_perturbation_results(perturbed_root_dir, baseline_dir, output_dir):
    """
    Parses perturbed CSVs and merges them with baseline CSVs.
    Outputs one aggregated CSV per algorithm.
    """
    # 1. Map old identifiers to your new preferred names
    old_to_new_map = {
        'western_us_power_grid': 'power',
        'western_us_power_grid_config': 'power_conf',
        'western_us_power_grid_er': 'power_er',
        'western_us_power_grid_sbm': 'power_sbm',
        'chloe_ppi_lcc_2026_02_23': 'ppi',
        'chloe_ppi_lcc_2026_02_23_config': 'ppi_conf',
        'chloe_ppi_lcc_2026_02_23_er': 'ppi_er',
        'chloe_ppi_lcc_2026_02_23_sbm': 'ppi_sbm',
        'ca-AstroPH_gcc': 'astro',
        'ca-AstroPH_gcc_config': 'astro_conf',
        'ca-AstroPH_gcc_er': 'astro_er',
        'ca-AstroPH_gcc_sbm': 'astro_sbm',
        'wiki-Vote_gcc': 'wiki',
        'wiki-Vote_gcc_config': 'wiki_conf',
        'wiki-Vote_gcc_er': 'wiki_er',
        'wiki-Vote_gcc_sbm': 'wiki_sbm'
    }
    
    # Sort keys by length descending so we don't accidentally match "power" inside "power_config"
    old_net_keys = sorted(old_to_new_map.keys(), key=len, reverse=True)
    
    algorithms = ['infomap', 'leiden', 'louvain', 'label_propagation']
    
    os.makedirs(output_dir, exist_ok=True)
    
    for alg in algorithms:
        print(f"Processing algorithm: {alg}...")
        all_rows = []
        
        # ---------------------------------------------------------
        # PART A: Process Baseline Data
        # ---------------------------------------------------------
        # Assuming your baseline CSVs are named something like "baseline_infomap.csv"
        # Adjust the filename pattern to match how you actually saved them
        baseline_csv = os.path.join(baseline_dir, f"baseline_community_stats_{alg}.csv") # Update if named differently
        
        if os.path.exists(baseline_csv):
            df_base = pd.read_csv(baseline_csv)
            for _, row in df_base.iterrows():
                # Get the new nice name (assuming baseline already has new names, if not, map it)
                net_name = row['network'] 
                
                all_rows.append({
                    'base_network_name': net_name,
                    'target': 'baseline',
                    'type_of_noise': 'none',
                    'level': 0.0,
                    'within_ari_mean': row.get('ari_mean', np.nan),
                    'within_ari_std': row.get('ari_std', 0.0),
                    'vs_baseline_ari_mean': 1.0, # Baseline compared to itself is exactly 1
                    'vs_baseline_ari_std': 0.0,
                    'n_communities_mean': row.get('num_comms_mean', np.nan),
                    'n_communities_std': row.get('num_comms_std', 0.0)
                })
        else:
            print(f"  Warning: Baseline CSV not found for {alg} at {baseline_csv}")
            
        # ---------------------------------------------------------
        # PART B: Process Perturbed Data
        # ---------------------------------------------------------
        # Search for all CSVs inside this algorithm's folder
        search_path = os.path.join(perturbed_root_dir, alg, '**', '*.csv')
        csv_files = glob.glob(search_path, recursive=True)
        
        for file_path in csv_files:
            filename = os.path.basename(file_path).replace('.csv', '')
            
            # 1. Identify the network by matching the old keys
            base_network_name = "unknown"
            for old_key in old_net_keys:
                if filename.startswith(old_key):
                    base_network_name = old_to_new_map[old_key]
                    break
                    
            if base_network_name == "unknown":
                continue # Skip files that don't match our known networks
                
            # 2. Extract metadata from the rest of the filename
            # e.g., ca-AstroPH_gcc_targeted_periphery_addition_noise_0p1
            target = 'random'
            if 'hub' in filename: target = 'hub'
            elif 'periphery' in filename: target = 'periphery'
            
            operation = 'addition' if 'add' in filename else 'removal'
            
            # Extract noise level (convert '0p1' to '0.1' or '50' to '50.0')
            try:
                noise_str = filename.split('noise_')[-1]
                level = float(noise_str.replace('p', '.'))
            except ValueError:
                level = np.nan
                
            # 3. Read the 100 repeats and aggregate
            try:
                df_pert = pd.read_csv(file_path)
                
                # We take the mean of the means, and the std of the means
                all_rows.append({
                    'base_network_name': base_network_name,
                    'target': target,
                    'type_of_noise': operation,
                    'level': level,
                    'within_ari_mean': df_pert['within_ari_mean'].mean(),
                    'within_ari_std': df_pert['within_ari_mean'].std(),
                    'vs_baseline_ari_mean': df_pert['vs_baseline_ari_mean'].mean(),
                    'vs_baseline_ari_std': df_pert['vs_baseline_ari_mean'].std(),
                    'n_communities_mean': df_pert['mean_n_communities'].mean(),
                    'n_communities_std': df_pert['mean_n_communities'].std()
                })
            except Exception as e:
                print(f"  Error reading {filename}: {e}")
                
        # ---------------------------------------------------------
        # PART C: Export Algorithm to CSV
        # ---------------------------------------------------------
        if all_rows:
            df_out = pd.DataFrame(all_rows)
            # Sort nicely so baseline is first, then by target, operation, and level
            df_out = df_out.sort_values(by=['base_network_name', 'target', 'type_of_noise', 'level'])
            
            out_file = os.path.join(output_dir, f"{alg}_aggregated.csv")
            df_out.to_csv(out_file, index=False)
            print(f"  -> Saved {len(df_out)} aggregated rows to {out_file}")

# --- Execution ---
# You need to update these paths to point to your actual folders!
aggregate_perturbation_results(
    perturbed_root_dir='./outputs/local_structure/recovery_perturbed', 
    baseline_dir='./outputs/local_structure/overview_csvs/baseline', 
    output_dir='./outputs/local_structure/overview_csvs/perturbed'
)