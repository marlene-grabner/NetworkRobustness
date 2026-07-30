import os
import glob
import pandas as pd
import numpy as np

def aggregate_local_neighborhood_results(parquet_dir, output_dir):
    """
    Reads all .parquet files in a directory and aggregates the local neighborhood 
    metrics into two CSVs. Groups dynamic maximum 'k' values together and 
    injects perfect baseline (0 noise) rows.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load all parquet files into a single DataFrame
    search_path = os.path.join(parquet_dir, '**', '*.parquet')
    parquet_files = glob.glob(search_path, recursive=True)
    
    if not parquet_files:
        print(f"No .parquet files found in {parquet_dir}")
        return
        
    print(f"Loading {len(parquet_files)} parquet files...")
    df_list = [pd.read_parquet(f) for f in parquet_files]
    df_all = pd.concat(df_list, ignore_index=True)
    
    # 2. Categorize the 'k' values
    fixed_ks = [10, 25, 50, 100]
    df_all['k_category'] = df_all['k'].apply(lambda x: str(int(x)) if x in fixed_ks else 'max')
    
    metrics = [
        'k', 'jaccard', 'precision', 'recall', 'f1', 
        'auroc', 'auprc', 'rank_zero_base', 'rank_zero_pert'
    ]
    
    # --- HELPER FUNCTION: Generate Perfect Baselines ---
    def generate_baseline_df(df_agg, group_cols):
        """Creates baseline rows with perfect metrics (1.0) and proper k values."""
        # Get unique combinations of the groups and their base network max ranks
        base_df = df_agg[group_cols + ['rank_zero_base_mean', 'rank_zero_base_std']].drop_duplicates().copy()
        
        base_df['noise_level'] = 0.0
        base_df['perturbation_type'] = 'baseline'
        base_df['modification_type'] = 'none'
        
        # Perfect scores for a baseline comparison
        perfect_metrics = ['jaccard', 'precision', 'recall', 'f1', 'auroc', 'auprc']
        for m in perfect_metrics:
            base_df[f'{m}_mean'] = 1.0
            base_df[f'{m}_std'] = 0.0
            
        # Determine the k_mean and k_std for the baseline
        # If it's 'max', use rank_zero_base. If it's fixed (e.g., 50), use that integer.
        base_df['k_mean'] = base_df.apply(
            lambda r: r['rank_zero_base_mean'] if r['k_category'] == 'max' else float(r['k_category']), axis=1
        )
        base_df['k_std'] = base_df.apply(
            lambda r: r['rank_zero_base_std'] if r['k_category'] == 'max' else 0.0, axis=1
        )
        
        # For a baseline graph, the perturbed rank is identical to the base rank
        base_df['rank_zero_pert_mean'] = base_df['rank_zero_base_mean']
        base_df['rank_zero_pert_std'] = base_df['rank_zero_base_std']
        
        return base_df

    # =====================================================================
    # 3. Aggregation 1: Grouped BY Seed 
    # =====================================================================
    print("Aggregating data (split by seed)...")
    group_cols_with_seed = [
        'network', 'perturbation_type', 'modification_type', 
        'noise_level', 'algorithm', 'k_category', 'seed_id'
    ]
    
    df_seed = df_all.groupby(group_cols_with_seed)[metrics].agg(['mean', 'std']).reset_index()
    
    # Flatten columns
    df_seed.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_seed.columns.values]
    
    # Inject Baselines
    baseline_seed_df = generate_baseline_df(df_seed, ['network', 'algorithm', 'k_category', 'seed_id'])
    df_seed = pd.concat([df_seed, baseline_seed_df], ignore_index=True)
    
    out_file_seed = os.path.join(output_dir, "local_results_by_seed.csv")
    df_seed.to_csv(out_file_seed, index=False)
    print(f"Saved seed-level aggregation to {out_file_seed}")
    
    # =====================================================================
    # 4. Aggregation 2: Grouped WITHOUT Seed
    # =====================================================================
    print("Aggregating data (ignoring seeds)...")
    group_cols_no_seed = [
        'network', 'perturbation_type', 'modification_type', 
        'noise_level', 'algorithm', 'k_category'
    ]
    
    df_no_seed = df_all.groupby(group_cols_no_seed)[metrics].agg(['mean', 'std']).reset_index()
    
    # Flatten columns
    df_no_seed.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_no_seed.columns.values]
    
    # Inject Baselines
    baseline_no_seed_df = generate_baseline_df(df_no_seed, ['network', 'algorithm', 'k_category'])
    df_no_seed = pd.concat([df_no_seed, baseline_no_seed_df], ignore_index=True)
    
    out_file_no_seed = os.path.join(output_dir, "local_results_aggregated.csv")
    df_no_seed.to_csv(out_file_no_seed, index=False)
    print(f"Saved global aggregation to {out_file_no_seed}")

# --- Execution ---
# aggregate_local_neighborhood_results('./local_results_parquets', './aggregated_results')

# --- Execution ---
# Update these paths to point to your actual folders
aggregate_local_neighborhood_results(
    parquet_dir='./outputs/seed_expansion/expansion/perturbed_metrics', 
    output_dir='./outputs/seed_expansion/expansion/perturbed_summary_csvs'
)