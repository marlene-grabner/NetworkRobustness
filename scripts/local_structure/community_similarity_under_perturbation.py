#%%
import argparse
import itertools
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from NoiseEffect.CommunityDetection.utils import convertPartitionToLabels, getMetrics
from NoiseEffect.CommunityDetection.detection_algorithms import (
    leidenAlgorithmPartioning,
    infomapAlgorithmPartioning,
    louvainPartioning,
    labelPropagationPartitioning,
)
from NoiseEffect.CommunityDetection.compare_perturbed_with_baseline_by_claude_idk_if_good import load_parquet_as_graphs, process_one_network

#%%

NOISE_TYPES = ["noise_type_a", "noise_type_b", "noise_type_c"]  # replace with yours
N_NOISE_LEVELS = 20
SEEDS = list(range(5))  # 5 seeds → 10 within-pairs; tune to your time budget

"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id",      type=int, required=True)
    parser.add_argument("--algo",         type=str, required=True)
    parser.add_argument("--graph-id",     type=int, required=True,
                        help="Which baseline graph this noise was applied to")
    parser.add_argument("--parquet-dir",  type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir",   type=Path, required=True)
    parser.add_argument("--n-jobs",       type=int, default=-1)
    args = parser.parse_args()

    # Decode task_id → noise_type, noise_level_idx
    noise_type      = NOISE_TYPES[args.task_id // N_NOISE_LEVELS]
    noise_level_idx = args.task_id % N_NOISE_LEVELS
    # Replace with your actual noise level values
    noise_level     = noise_level_idx / N_NOISE_LEVELS

    # Load baseline label matrix
    baseline_path = args.baseline_dir / f"baseline_{args.graph_id}_{args.algo}.npz"
    baseline_labels = np.load(baseline_path)["labels"]  # (k_baseline, n_nodes)

    # Find and load the right parquet
    parquet_path = (
        args.parquet_dir
        / f"graph_{args.graph_id}"
        / noise_type
        / f"noise_level_{noise_level_idx:02d}.parquet"
    )
    networks = load_parquet_as_graphs(parquet_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output_dir
        / f"graph_{args.graph_id}_{noise_type}_level{noise_level_idx:02d}_{args.algo}.parquet"
    )
    if out_path.exists():
        print(f"Already done: {out_path}, skipping.")
        return

    # Parallel over 100 networks in this parquet
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(process_one_network)(
            repeat_id, edges, n_nodes,
            args.algo, SEEDS, baseline_labels,
            args.graph_id, noise_type, noise_level,
        )
        for repeat_id, edges, n_nodes in networks
    )

    pd.DataFrame(results).to_parquet(out_path, index=False)
    print(f"Done → {out_path}")
"""

#%%
if __name__ == "__main__":
    parquet_path = '../../../data/perturbed_networks/western_us_power_grid/perturbed_periphery_target/western_us_power_grid_targeted_periphery_addition_noise_0p2.parquet'
    networks = load_parquet_as_graphs(parquet_path)
    x = process_one_network(
        repeat_id=0,
        edges=networks[0][1],
        n_nodes=networks[0][2],
        algo='leiden',
        seeds=SEEDS,
        baseline_labels=np.random.randint(0, 5, size=(len(SEEDS), networks[0][2])),  # Dummy baseline labels for testing
        graph_id=0,
        noise_type='noise_type_a',
        noise_level=0.1,
    )
    print(x)
# %%

# %%
