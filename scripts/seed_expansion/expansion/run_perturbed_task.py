"""
One SLURM array task = one row of results/job_tables/perturbed_jobs.csv
= one (network, perturbation_type, noise_level) parquet file, which itself
holds all 100 repeats.

For each repeat and each of the 4 algorithms:
    - build adjacency from that repeat's edgelist (reusing the network's
      cached NodeIndex, so it aligns with the baseline ranking)
    - compute the algorithm's score vector
    - compare it to the baseline score vector for that network/algorithm
      (top-k jaccard/precision/recall/f1 + AUROC/AUPRC per k)
    - discard the full score vector -- only the metric rows are kept

All metric rows for the task are written to a single parquet file, so a
960-task array produces 960 files (not 960*100*4). Run
notebooks/aggregate_perturbed_metrics.py afterwards to concatenate them.

Usage (called by slurm/perturbed_array.sbatch):
    python notebooks/run_perturbed_task.py --job-row 3 --k-list 10,25,50,100
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import NoiseEffect.SeedExpansion.Expansion.io_helper as io
from NoiseEffect.SeedExpansion.Expansion import run_algorithm
from NoiseEffect.SeedExpansion.Expansion.comparing import compare_rankings

JOB_TABLE = Path("./slurm_scripts/seed_expansion/expansion/jobs/perturbed_jobs.csv")
SEEDS_CSV = Path("./outputs/seed_expansion/synthetic_seeds/synthetic_seeds_by_bsf.csv")
NODE_INDEX_DIR = Path("./outputs/seed_expansion/expansion/node_index")
BASELINE_RANKING_DIR = Path("./outputs/seed_expansion/expansion/baseline_rankings")
OUT_DIR = Path("./outputs/seed_expansion/expansion/perturbed_metrics")

ALGORITHMS = ["rwr_row", "rwr_sym", "diamond", "first_neighbors"]


def main():

    ap = argparse.ArgumentParser()
    # 1. Add the new argument here
    ap.add_argument("--job-table", type=str, required=True,
                    help="Path to the perturbed jobs CSV file")
    ap.add_argument("--job-row", type=int, required=True)
    ap.add_argument("--k-list", type=str, default="10,25,50,100",
                    help="comma-separated list of k values for top-k metrics")
    args = ap.parse_args()
    k_list = [int(k) for k in args.k_list.split(",")]

    # 2. Use the argument here
    jobs = pd.read_csv(args.job_table)
    job = jobs.iloc[args.job_row - 1]

    network = job["network"]
    perturbation_type = job["perturbation_type"]
    modification_type = job["modification_type"]
    noise_level = job["noise_level"]
    parquet_path = job["parquet_path"]

    t0 = time.time()

    index = io.load_node_index(NODE_INDEX_DIR / f"{network}.pkl")
    seed_table = io.load_seed_table(SEEDS_CSV)
    
    # 1. Extract the separated list of (seed_id, seed_idx) pairs for this network
    network_seeds = list(io.iter_network_seeds(seed_table, network, index))

    # Baseline scores per algorithm, loaded once for the whole task
    baseline_scores = {}
    for algo in ALGORITHMS:
        baseline_scores[algo] = {}
        b_path = BASELINE_RANKING_DIR / f"{network}_{algo}.parquet"
        
        if b_path.exists():
            df_b = pd.read_parquet(b_path)
            for seed_id, sub_df in df_b.groupby("seed_id"):
                sub_df = sub_df.sort_values("node_idx")
                baseline_scores[algo][seed_id] = sub_df["score"].values
        else:
            print(f"Warning: Baseline file missing for {network} + {algo}")
            
    rows = []
    n_dropped_total = 0
    n_repeats = 0

    # 2. Stream the perturbed network repetitions
    for repeat_id, adj, n_dropped in io.iter_perturbed_repeats(parquet_path, index):
        n_repeats += 1
        n_dropped_total += n_dropped

        # 3. Loop through each separate seed sequence
        for seed_id, seed_idx in network_seeds:
            
            # 4. Evaluate each algorithm independently for this specific seed configuration
            for algo in ALGORITHMS:
                # Ensure we actually have the matching baseline array cached
                if seed_id not in baseline_scores[algo]:
                    continue
                
                # Compute expansion metrics on the perturbed graph for this seed
                perturbed_scores = run_algorithm(algo, adj, seed_idx)
                
                # Pull the matching isolated baseline numpy array out of our dictionary
                b_scores = baseline_scores[algo][seed_id]
                
                # Calculate metrics (Jaccard, Precision, Recall, F1, AUROC, AUPRC)
                metric_rows = compare_rankings(b_scores, perturbed_scores, k_list)
                
                for r in metric_rows:
                    r.update({
                        "network": network,
                        "perturbation_type": perturbation_type,
                        "modification_type": modification_type,
                        "noise_level": noise_level,
                        "repeat": repeat_id,
                        "algorithm": algo,
                        "seed_id": seed_id,  # CRITICAL: Track the seed ID alongside your metrics!
                    })
                    rows.append(r)

    # Save out the compiled metrics 
    noise_level_str = str(noise_level).replace(".", "p")
    out_path = OUT_DIR / f"{network}_{perturbation_type}_{modification_type}_{noise_level_str}.parquet"
    io.append_metrics_rows(rows, out_path)

    print(f"[{network}/{perturbation_type}/{modification_type}/noise={noise_level}] "
          f"{n_repeats} repeats x {len(network_seeds)} seeds x {len(ALGORITHMS)} algos x {len(k_list)} k-values "
          f"= {len(rows)} metric rows -> {out_path} "
          f"({n_dropped_total} edges dropped total) ({time.time() - t0:.1f}s)")

if __name__ == "__main__":
    main()

