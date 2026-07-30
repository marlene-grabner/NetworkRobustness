import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import NoiseEffect.SeedExpansion.Expansion.io_helper as io
from NoiseEffect.SeedExpansion.Expansion import run_algorithm

JOB_TABLE = Path("./slurm_scripts/seed_expansion/expansion/jobs/baseline_jobs.csv")
SEEDS_CSV = Path("./outputs/seed_expansion/synthetic_seeds/synthetic_seeds_by_bsf.csv")
NODE_INDEX_DIR = Path("./outputs/seed_expansion/expansion/node_index")
RANKING_DIR = Path("./outputs/seed_expansion/expansion/baseline_rankings")

def main():
    # parsing the SLURM_ARRAY_TASK_ID:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-row", type=int, required=True)
    args = ap.parse_args()
    job_row = args.job_row
    
    jobs = pd.read_csv(JOB_TABLE)

    job = jobs.iloc[job_row - 1]
    
    network, edgelist_path, algorithm = job["network"], job["edgelist_path"], job["algorithm"]
    print(f"Processing baseline network: {network} using {algorithm}")
    edge_df = io.load_baseline_edgelist(edgelist_path)
    
    # Track node indices across experiments
    index_path = NODE_INDEX_DIR / f"{network}.pkl"
    if index_path.exists():
        index = io.load_node_index(index_path)
    else:
        index = io.build_node_index(network, edge_df)
        io.save_node_index(index, index_path)
        
    adj, _ = io.edges_to_sparse(edge_df, index)
    seed_table = io.load_seed_table(SEEDS_CSV)
    network_seeds = list(io.iter_network_seeds(seed_table, network, index))

    all_seed_rankings = []

    for seed_id, seed_idx in network_seeds:
        t0 = time.time()
        scores = run_algorithm(algorithm, adj, seed_idx)
        
        # Generate the ranking order
        order = np.argsort(-scores, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)
        
        # Build a temporary dataframe for this seed
        df_seed = pd.DataFrame({
            "node_idx": np.arange(index.n_nodes),
            "score": scores,
            "rank": rank,
            "seed_id": seed_id  # This keeps them distinct!
        })
        all_seed_rankings.append(df_seed)
        print(f"  Calculated baseline for {seed_id} ({time.time() - t0:.2f}s)")

    # Consolidate and save to a single file for this network + algorithm pair
    if all_seed_rankings:
        consolidated_df = pd.concat(all_seed_rankings, ignore_index=True)
        
        # Output structure: baseline_rankings/power_rwr_row.parquet
        out_path = RANKING_DIR / f"{network}_{algorithm}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        consolidated_df.to_parquet(out_path, index=False)
        print(f"Saved all seed baselines to consolidated file: {out_path}")

if __name__ == "__main__":
    main()
