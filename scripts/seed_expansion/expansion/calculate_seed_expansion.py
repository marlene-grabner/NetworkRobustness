"""
Seed expansion (rwr_row, rwr_sym, diamond, first_neighbors) on all perturbed
networks, with the seed sets of
scripts/seed_expansion/synthetic_seed_generation/synthetic_seed_generation.py
(src/NoiseEffect/SeedExpansion/Expansion).

Step 1, once: seed expansion on all unperturbed networks
  uv run python scripts/seed_expansion/expansion/calculate_seed_expansion.py baselines [n_workers]
Step 2, as a job array: task i of n takes every n-th perturbed file
  uv run python scripts/seed_expansion/expansion/calculate_seed_expansion.py <i> <n> [n_workers]

Output in outputs/seed_expansion/results/{network}/{model}/:
  baseline_tops.npz             top-100 nodes per algorithm, replicate and seed set (step 1)
  baseline.csv                  recovery of the hidden module nodes on the
                                unperturbed networks, per replicate, seed set, algorithm and k
  {noise_type}_{level}.parquet  one row per perturbed network, seed set, algorithm and k:
      network, model, noise_type, target, action, level, replicate, repeat,
      set_id, seed_type, algorithm, k,
      recovery: recovery_recall, recovery_precision, recovery_r_precision,
                recovery_auroc, recovery_auprc
      stability (vs the baseline top-k): jaccard, precision, recall, f1, auroc, auprc
      n_tied_outside, n_valid
  Join with the seed table on network, model, replicate and set_id for the
  details of a seed set (degree percentile, community size, ...).

Files that exist already are skipped, so an interrupted run can be restarted.
aggregate_seed_expansion.py combines everything into one table.
"""

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from NoiseEffect.SeedExpansion import Expansion as se
from NoiseEffect.utils.perturbed_io import load_node_index

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "perturbed_networks"))
from generating_perturbed_networks import all_units  # noqa: E402

SEEDS_CSV = "./outputs/seed_expansion/synthetic_seeds/seed_sets.csv"
RESULTS_DIR = "./outputs/seed_expansion/results/"


def unit_info(unit, units):
    """network, model, empirical edgelist, baseline edgelists, perturbed and results folder"""
    baselines, perturbed_folder, _ = units[unit]
    perturbed_folder = os.path.normpath(perturbed_folder)
    network, model = perturbed_folder.split(os.sep)[-2:]
    return {
        "network": network,
        "model": model,
        # all models of a network share the node set of the empirical network
        "empirical_edgelist": units[network][0][0][0],
        "baseline_edgelists": [edgelist for edgelist, _ in baselines],
        "perturbed_folder": perturbed_folder,
        "results_folder": os.path.join(RESULTS_DIR, network, model),
    }


def load_inputs(info):
    node_to_idx = load_node_index(info["empirical_edgelist"])
    seed_table = pd.read_csv(SEEDS_CSV, dtype={"seed_nodes": str, "hidden_nodes": str})
    seed_sets = se.seed_sets_of_unit(seed_table, info["network"], info["model"], node_to_idx)
    return node_to_idx, seed_sets


def calculate_baselines(info):
    tops_path = os.path.join(info["results_folder"], "baseline_tops.npz")
    if os.path.exists(tops_path):
        return f"exists already: {tops_path}"
    try:
        os.makedirs(info["results_folder"], exist_ok=True)
        node_to_idx, seed_sets = load_inputs(info)
        results, tops = se.baseline_seed_expansion(
            info["baseline_edgelists"], node_to_idx, seed_sets
        )
    except Exception as error:
        return f"FAILED {info['network']}/{info['model']}: {type(error).__name__}: {error}"
    results.insert(0, "network", info["network"])
    results.insert(1, "model", info["model"])
    results.to_csv(os.path.join(info["results_folder"], "baseline.csv"), index=False)
    # written under another name first, so a killed job leaves no broken file
    np.savez_compressed(tops_path + ".tmp.npz", **tops)
    os.replace(tops_path + ".tmp.npz", tops_path)
    return f"done: {tops_path}"


def calculate_file(path, info):
    result_path = os.path.join(info["results_folder"], os.path.basename(path))
    if os.path.exists(result_path):
        return f"exists already: {result_path}"
    try:
        node_to_idx, seed_sets = load_inputs(info)
        with np.load(os.path.join(info["results_folder"], "baseline_tops.npz")) as f:
            baseline_tops = dict(f)
        results = se.seed_expansion_of_file(path, node_to_idx, seed_sets, baseline_tops)
    except Exception as error:
        return f"FAILED {path}: {type(error).__name__}: {error}"
    results.to_parquet(result_path + ".tmp", index=False)
    os.replace(result_path + ".tmp", result_path)
    return f"done: {result_path} ({len(results)} rows)"


if __name__ == "__main__":
    if not os.path.exists(SEEDS_CSV):
        sys.exit(f"{SEEDS_CSV} is missing, run synthetic_seed_generation.py first.")
    units = all_units()
    infos = [unit_info(unit, units) for unit in units]

    if sys.argv[1] == "baselines":
        n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        failed = 0
        with ProcessPoolExecutor(n_workers) as pool:
            for message in pool.map(calculate_baselines, infos):
                failed += message.startswith("FAILED")
                print(message, flush=True)
        if failed:
            sys.exit(f"{failed} of {len(infos)} baselines failed")
        sys.exit()

    task, n_tasks = int(sys.argv[1]), int(sys.argv[2])
    missing = [
        info["results_folder"]
        for info in infos
        if not os.path.exists(os.path.join(info["results_folder"], "baseline_tops.npz"))
    ]
    if missing:
        sys.exit(
            f"The baseline results of {len(missing)} of {len(infos)} units are "
            f"missing, e.g. in {missing[0]}. Run the 'baselines' step first."
        )
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    jobs = [
        (path, info)
        for info in infos
        for path in sorted(glob.glob(os.path.join(info["perturbed_folder"], "*.parquet")))
    ]
    print(f"task {task} of {n_tasks}: {len(jobs[task::n_tasks])} of {len(jobs)} files")
    failed = 0
    with ProcessPoolExecutor(n_workers) as pool:
        for message in pool.map(calculate_file, *zip(*jobs[task::n_tasks])):
            failed += message.startswith("FAILED")
            print(message, flush=True)
    if failed:
        sys.exit(f"task {task}: {failed} files failed")
