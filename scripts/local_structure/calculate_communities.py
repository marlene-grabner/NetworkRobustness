"""
Community detection on all perturbed networks (leiden, infomap and label
propagation, 10 seeds each; src/NoiseEffect/CommunityDetection).

Step 1, once: the partitions of all unperturbed networks
  uv run python scripts/local_structure/calculate_communities.py baselines [n_workers]
Step 2, as a job array: task i of n takes every n-th perturbed file
  uv run python scripts/local_structure/calculate_communities.py <i> <n> [n_workers]

Output in outputs/local_structure/communities/{network}/{model}/:
  baseline_partitions.npz       labels of the unperturbed networks (step 1)
  baseline.csv                  their seed-to-seed ARI and collapse statistics
  {noise_type}_{level}.parquet  one row per perturbed network and algorithm:
      network, model, noise_type, target, action, level, replicate, repeat,
      algorithm, ARI to the baseline, ARI between seeds, collapse statistics

Files that exist already are skipped, so an interrupted run can be restarted.
aggregate_communities.py combines everything into one table.
"""

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from NoiseEffect import CommunityDetection as cd
from NoiseEffect.utils.perturbed_io import load_node_index

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "perturbed_networks"))
from generating_perturbed_networks import all_units  # noqa: E402

RESULTS_DIR = "./outputs/local_structure/communities/"


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


def calculate_baselines(info):
    partitions_path = os.path.join(info["results_folder"], "baseline_partitions.npz")
    if os.path.exists(partitions_path):
        return f"exists already: {partitions_path}"
    os.makedirs(info["results_folder"], exist_ok=True)
    node_to_idx = load_node_index(info["empirical_edgelist"])
    partitions = cd.baseline_partitions(info["baseline_edgelists"], node_to_idx)
    results = cd.community_properties_of_baselines(partitions)
    results.insert(0, "network", info["network"])
    results.insert(1, "model", info["model"])
    results.to_csv(os.path.join(info["results_folder"], "baseline.csv"), index=False)
    # written under another name first, so a killed job leaves no broken file
    np.savez_compressed(partitions_path + ".tmp.npz", **partitions)
    os.replace(partitions_path + ".tmp.npz", partitions_path)
    return f"done: {partitions_path}"


def calculate_file(path, info):
    result_path = os.path.join(info["results_folder"], os.path.basename(path))
    if os.path.exists(result_path):
        return f"exists already: {result_path}"
    try:
        node_to_idx = load_node_index(info["empirical_edgelist"])
        with np.load(
            os.path.join(info["results_folder"], "baseline_partitions.npz")
        ) as f:
            baselines = dict(f)
        results = cd.community_properties_of_file(path, node_to_idx, baselines)
    except Exception as error:
        return f"FAILED {path}: {type(error).__name__}: {error}"
    results.to_parquet(result_path + ".tmp", index=False)
    os.replace(result_path + ".tmp", result_path)
    return f"done: {result_path} ({len(results)} rows)"


if __name__ == "__main__":
    units = all_units()
    infos = [unit_info(unit, units) for unit in units]

    if sys.argv[1] == "baselines":
        n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        with ProcessPoolExecutor(n_workers) as pool:
            for message in pool.map(calculate_baselines, infos):
                print(message, flush=True)
        sys.exit()

    task, n_tasks = int(sys.argv[1]), int(sys.argv[2])
    missing = [
        info["results_folder"]
        for info in infos
        if not os.path.exists(
            os.path.join(info["results_folder"], "baseline_partitions.npz")
        )
    ]
    if missing:
        sys.exit(
            f"The baseline partitions of {len(missing)} of {len(infos)} units are "
            f"missing, e.g. in {missing[0]}. Run the 'baselines' step first."
        )
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    jobs = [
        (path, info)
        for info in infos
        for path in sorted(
            glob.glob(os.path.join(info["perturbed_folder"], "*.parquet"))
        )
    ]
    print(f"task {task} of {n_tasks}: {len(jobs[task::n_tasks])} of {len(jobs)} files")
    failed = 0
    with ProcessPoolExecutor(n_workers) as pool:
        for message in pool.map(calculate_file, *zip(*jobs[task::n_tasks])):
            failed += message.startswith("FAILED")
            print(message, flush=True)
    if failed:
        sys.exit(f"task {task}: {failed} files failed")
