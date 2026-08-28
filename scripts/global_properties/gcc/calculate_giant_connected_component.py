from NoiseEffect import calculate_singletons_and_gcc
from pathlib import Path
import os
import igraph as ig
import pandas as pd


baseline_tsvs = {
    #"ppi": "data/baseline_networks/old_tsv_networks/chloe_ppi_lcc_2026_02_23.tsv",
    #"astro": "data/baseline_networks/old_tsv_networks/ca-AstroPh_gcc.tsv",
    #"power":"data/baseline_networks/old_tsv_networks/western_us_power_grid.tsv",
    #"wiki": "data/baseline_networks/old_tsv_networks/wiki-Vote_gcc.tsv",
    #"ppi_er": "data/baseline_networks/old_tsv_networks/null_models/chloe_ppi_erdos_renyi.tsv",
    #"ppi_conf": "data/baseline_networks/old_tsv_networks/null_models/chloe_ppi_configuration_model.tsv",
    "ppi_sbm": "data/baseline_networks/old_tsv_networks/null_models/chloe_ppi_sbm.tsv",
    #"astro_er": "data/baseline_networks/old_tsv_networks/null_models/ca-AstroPh_erdos_renyi.tsv",
    #"astro_conf": "data/baseline_networks/old_tsv_networks/null_models/ca-AstroPh_configuration_model.tsv",
    "astro_sbm": "data/baseline_networks/old_tsv_networks/null_models/ca-AstroPh_sbm.tsv",
    "power_er": "data/baseline_networks/old_tsv_networks/null_models/western_us_power_grid_erdos_renyi.tsv",
    #"power_conf": "data/baseline_networks/old_tsv_networks/null_models/western_us_power_grid_configuration_model.tsv",
    "power_sbm": "data/baseline_networks/old_tsv_networks/null_models/western_us_power_grid_sbm.tsv",
    #"wiki_er": "data/baseline_networks/old_tsv_networks/null_models/wiki-Vote_erdos_renyi.tsv",
    #"wiki_conf": "data/baseline_networks/old_tsv_networks/null_models/wiki-Vote_configuration_model.tsv",
    "wiki_sbm": "data/baseline_networks/old_tsv_networks/null_models/wiki-Vote_sbm.tsv"
}

perturbed_folders = {
    #"ppi": "data/perturbed_networks/ppi",
    #"astro": "data/perturbed_networks/astro",
    #"power":"data/perturbed_networks/power",
    #"wiki": "data/perturbed_networks/wiki",
    #"ppi_er": "data/perturbed_networks/ppi_er",
    #"ppi_conf": "data/perturbed_networks/ppi_conf",
    "ppi_sbm": "data/perturbed_networks/ppi_sbm",
    #"astro_er": "data/perturbed_networks/astro_er",
    #"astro_conf": "data/perturbed_networks/astro_conf",
    "astro_sbm": "data/perturbed_networks/astro_sbm",
    "power_er": "data/perturbed_networks/power_er",
    #"power_conf": "data/perturbed_networks/power_conf",
    "power_sbm": "data/perturbed_networks/power_sbm",
    #"wiki_er": "data/perturbed_networks/wiki_er",
    #"wiki_conf": "data/perturbed_networks/wiki_conf",
    "wiki_sbm": "data/perturbed_networks/wiki_sbm"
}

#########################################################
# To integrate degree-0 nodes in total node count
# (lost on baseline in SBM and ER)
###########################################################

def _true_baseline_node_count(baseline_path):
    df_base = pd.read_csv(baseline_path, sep='\t', header=None, names=['source', 'target'])
    nodes = set(df_base['source']).union(set(df_base['target']))
    # union in isolated nodes from the sidecar, if one exists
    sidecar = baseline_path.rsplit('.', 1)[0] + '_isolated_nodes.csv'
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            content = f.read().strip()
        if content:
            nodes |= {n.strip() for n in content.split(',') if n.strip()}
    return len(nodes)


if __name__ == '__main__':
    # Grab the number of CPUs from Slurm, default to 4 if running locally
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

    # Location for results
    out_path = Path('./outputs/global_properties/gcc_singletons/gcc_singletons_per_network')
    out_path.mkdir(parents=True, exist_ok=True)
    
    for key in baseline_tsvs.keys():
        print(f"--- Starting analysis for network: {key} ---")
        
        all_dfs_for_key = [] # Reset the list for each new network
        baseline_tsv = baseline_tsvs[key]
        perturbed_networks_location = perturbed_folders[key]
        
        # ---------------------------------------------------------
        # CALCULATE BASELINE METRICS
        # ---------------------------------------------------------
        # Load baseline to get total nodes and calculate its GCC
        df_base = pd.read_csv(baseline_tsv, sep='\t', header=None, names=['source', 'target'])
        total_baseline_nodes = _true_baseline_node_count(baseline_tsv)
        
        # Build baseline igraph
        g_base = ig.Graph.TupleList(df_base.values.tolist(), directed=False)
        base_gcc_size = max(g_base.components().sizes()) if g_base.vcount() > 0 else 0
        
        baseline_result = pd.DataFrame([{
            'network_id': f"{key}_baseline",
            'num_singletons': 0, # Edgelists naturally contain no singletons
            'gcc': base_gcc_size / total_baseline_nodes,
            'perturbation_method': 'baseline'
        }])
        all_dfs_for_key.append(baseline_result)
        
        # ---------------------------------------------------------
        # CALCULATE PERTURBED METRICS
        # ---------------------------------------------------------
        for perturbation_method in os.listdir(perturbed_networks_location):
            perturbed_dir = os.path.join(perturbed_networks_location, perturbation_method)            
            df = calculate_singletons_and_gcc(
                baseline_path=baseline_tsv, 
                perturbed_dir=perturbed_dir, 
                max_workers=num_workers
            )
            df['perturbation_method'] = perturbation_method
            all_dfs_for_key.append(df)

        # ---------------------------------------------------------
        # SAVE CSV FOR THIS NETWORK KEY
        # ---------------------------------------------------------
        final_key_df = pd.concat(all_dfs_for_key, ignore_index=True)
        out_filename = out_path / f"gcc_singletons_{key}.csv"
        final_key_df.to_csv(out_filename, index=False)