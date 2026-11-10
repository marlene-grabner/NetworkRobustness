"""
Generates the perturbed networks for one network unit: an empirical network
(core or extended set) or one null model of a core network (all its
replicates go into the same files).

Output, one file per noise type and level (see NoiseNetworks/main.py):
  data/perturbed_networks_ensembles/{network}/{empirical|er|conf|sbm}/{noise_type}_{level}.parquet
  columns: source, target, replicate, repeat

Run from the repository root:
  uv run python scripts/perturbed_networks/generating_perturbed_networks.py <unit>
  uv run python scripts/perturbed_networks/generating_perturbed_networks.py --list
<unit> is a network name (e.g. huri, chicago_road) or a core network and null
model (e.g. huri_er, power_sbm). --list prints all units, one per line, e.g.
for a job array.
"""

import os
import sys

import pandas as pd

import NoiseEffect as na

BASELINE_DIR = "./data/baseline_networks/"
NULL_DIR = BASELINE_DIR + "null_models/"
OUT_DIR = "./data/perturbed_networks_ensembles/"

# Core networks get null models (astro.csv has its 59 self-loops removed,
# like its null models; the original is in baseline_networks/backup/)
CORE_NETWORKS = ["huri", "power", "astro", "wiki"]
NULL_MODELS = ["er", "conf", "sbm"]
N_NULL_REPLICATES = 10

REPEATS_EMPIRICAL = 25
REPEATS_NULL = 10

NOISE_LEVELS_REMOVED = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
NOISE_LEVELS_ADDED = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 1.50, 2.00]
NOISE_LEVELS = {
    "random_addition": NOISE_LEVELS_ADDED,
    "random_removal": NOISE_LEVELS_REMOVED,
    "hub_addition": NOISE_LEVELS_ADDED,
    "hub_removal": NOISE_LEVELS_REMOVED,
    "periphery_addition": NOISE_LEVELS_ADDED,
    "periphery_removal": NOISE_LEVELS_REMOVED,
    "betweenness_removal": NOISE_LEVELS_REMOVED,
    "triadic_closure_addition": NOISE_LEVELS_ADDED,
    "degree_preserving_rewiring": NOISE_LEVELS_REMOVED,  # fraction of edges rewired
}


def extended_set_networks():
    overview = pd.read_csv(BASELINE_DIR + "extended_set/extended_set_overview.csv")
    return overview.loc[overview.status == "ok", "key"].tolist()


def all_units():
    """unit -> (baselines, output folder, repeats)"""
    units = {}
    for network in CORE_NETWORKS:
        units[network] = (
            [(f"{BASELINE_DIR}{network}.csv", None)],
            f"{OUT_DIR}{network}/empirical",
            REPEATS_EMPIRICAL,
        )
        for model in NULL_MODELS:
            baselines = []
            for rep in range(N_NULL_REPLICATES):
                edgelist = f"{NULL_DIR}{model}_ensemble/{network}_{model}_rep{rep}.csv"
                sidecar = edgelist.replace(".csv", "_isolated_nodes.csv")  # ER only
                baselines.append(
                    (edgelist, sidecar if os.path.exists(sidecar) else None)
                )
            units[f"{network}_{model}"] = (
                baselines,
                f"{OUT_DIR}{network}/{model}",
                REPEATS_NULL,
            )
    for network in extended_set_networks():
        units[network] = (
            [(f"{BASELINE_DIR}extended_set/{network}.csv", None)],
            f"{OUT_DIR}{network}/empirical",
            REPEATS_EMPIRICAL,
        )
    return units


if __name__ == "__main__":
    units = all_units()
    if sys.argv[1] == "--list":
        print("\n".join(units))
        sys.exit()

    unit = sys.argv[1]
    baselines, folder, repeats = units[unit]
    print(f"{unit}: {len(baselines)} replicate(s) x {repeats} repeats -> {folder}")
    skipped = na.generatePerturbedNetworks(
        baselines=baselines,
        folder_to_save_perturbed=folder,
        noise_levels=NOISE_LEVELS,
        num_repeats=repeats,
    )
    if skipped:
        print(f"{unit}: skipped (not reachable): {skipped}")
