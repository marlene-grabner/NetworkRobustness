"""
Calculates the global properties of all perturbed networks of one unit: an
empirical network or one null model of a core network (the same units as in
scripts/perturbed_networks/generating_perturbed_networks.py).

Metrics (src/NoiseEffect/GlobalProperties): gcc and singletons, global
efficiency, transitivity, degree assortativity, k-core, largest adjacency
eigenvalue, centrality ranks (degree and PageRank).

Output, one row per network:
  outputs/global_properties/all_metrics/{network}/{model}/{noise_type}_{level}.parquet
    perturbed networks of one file
    columns: network, model, noise_type, target, action, level, replicate, repeat, metrics
  outputs/global_properties/all_metrics/{network}/{model}/baseline.csv
    unperturbed networks, columns: network, model, replicate, metrics

Files that exist already are skipped, so an interrupted run can be restarted.
aggregate_global_properties.py combines everything into one table.

Run from the repository root:
  uv run python scripts/global_properties/calculate_global_properties.py <unit> [n_workers]
"""

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from NoiseEffect import GlobalProperties as gp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "perturbed_networks"))
from generating_perturbed_networks import all_units  # noqa: E402

RESULTS_DIR = "./outputs/global_properties/all_metrics/"


def calculate_file(path, results_folder, empirical_edgelist, baseline_edgelists):
    result_path = os.path.join(results_folder, os.path.basename(path))
    if os.path.exists(result_path):
        return f"exists already: {result_path}"
    try:
        node_to_idx = gp.load_node_index(empirical_edgelist)
        results = gp.global_properties_of_file(path, node_to_idx, baseline_edgelists)
    except Exception as error:
        return f"FAILED {path}: {error!r}"
    # written under another name first, so a killed job leaves no broken file
    results.to_parquet(result_path + ".tmp", index=False)
    os.replace(result_path + ".tmp", result_path)
    return f"done: {result_path} ({len(results)} networks)"


def calculate_baselines(results_folder, network, model, empirical_edgelist, edgelists):
    result_path = os.path.join(results_folder, "baseline.csv")
    if os.path.exists(result_path):
        return
    node_to_idx = gp.load_node_index(empirical_edgelist)
    rows = [
        {
            "network": network,
            "model": model,
            "replicate": replicate,
            **gp.global_properties_of_baseline(edgelist, node_to_idx),
        }
        for replicate, edgelist in enumerate(edgelists)
    ]
    pd.DataFrame(rows).to_csv(result_path, index=False)


if __name__ == "__main__":
    unit = sys.argv[1]
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    units = all_units()
    baselines, perturbed_folder, _ = units[unit]
    perturbed_folder = os.path.normpath(perturbed_folder)
    network, model = perturbed_folder.split(os.sep)[-2:]
    # all models of a network share the node set of the empirical network
    empirical_edgelist = units[network][0][0][0]
    baseline_edgelists = [edgelist for edgelist, _ in baselines]
    results_folder = os.path.join(RESULTS_DIR, network, model)
    os.makedirs(results_folder, exist_ok=True)

    files = sorted(glob.glob(os.path.join(perturbed_folder, "*.parquet")))
    if not files:
        sys.exit(f"{unit}: no perturbed networks in {perturbed_folder}")
    print(f"{unit}: {len(files)} files, {n_workers} worker(s)", flush=True)

    calculate_baselines(
        results_folder, network, model, empirical_edgelist, baseline_edgelists
    )
    failed = 0
    with ProcessPoolExecutor(n_workers) as pool:
        jobs = [
            pool.submit(
                calculate_file,
                f,
                results_folder,
                empirical_edgelist,
                baseline_edgelists,
            )
            for f in files
        ]
        for job in jobs:
            message = job.result()
            failed += message.startswith("FAILED")
            print(message, flush=True)
    if failed:
        sys.exit(f"{unit}: {failed} of {len(files)} files failed")
