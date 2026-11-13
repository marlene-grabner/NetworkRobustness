"""
Synthetic seed sets for all network units (the same units as in
scripts/perturbed_networks/generating_perturbed_networks.py): every empirical
network and every replicate of every null model gets its own seed sets, made
on that unperturbed network (src/NoiseEffect/SeedExpansion/SyntheticSeeds).

Two seed types; every module is split at random into SEED_FRACTION seeds and
the hidden rest, which the algorithms should recover:
  bfs        N_BFS_SETS modules of BFS_MODULE_SIZE nodes per degree
             percentile (BFS around a random start node of about that
             degree); no two modules of a network with a Jaccard overlap
             above MAX_OVERLAP
  community  N_COMMUNITY_SETS whole Leiden communities with
             MIN_COMMUNITY_SIZE to MAX_COMMUNITY_SIZE nodes, drawn at random.
             The partition is the one of the community pipeline for its first
             seed (same algorithm, settings and seed;
             CommunityDetection.detect_partitions). Modularity communities are
             often much larger, so communities above MAX_COMMUNITY_SIZE are
             split again with Leiden on their own subgraph, recursively, until
             every piece is small enough; parent_community_id is the community
             of the pipeline partition a piece lies in, split_depth the number
             of splits (0 = an original community)

Output: outputs/seed_expansion/synthetic_seeds/seed_sets.csv
  network, model, replicate, set_id, seed_type, n_seeds, n_hidden,
  target_percentile, target_degree, start_node, start_degree, max_overlap (bfs),
  community_id, community_size, parent_community_id, split_depth (community),
  seed_nodes, hidden_nodes (node labels joined by ";")
set_id is unique per network, model and replicate.

Run from the repository root:
  uv run python scripts/seed_expansion/synthetic_seed_generation/synthetic_seed_generation.py [n_workers]
"""

import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from NoiseEffect.CommunityDetection import SEEDS, detect_partitions
from NoiseEffect.SeedExpansion.Expansion import edges_to_adjacency
from NoiseEffect.SeedExpansion.SyntheticSeeds import (
    bfs_seed_sets,
    community_seed_sets,
    split_large_communities,
)
from NoiseEffect.utils.perturbed_io import load_node_index, read_baseline_edges

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "perturbed_networks"))
from generating_perturbed_networks import all_units  # noqa: E402

OUT_PATH = "./outputs/seed_expansion/synthetic_seeds/seed_sets.csv"

SEED_FRACTION = 0.8  # of every module; the rest is hidden

# bfs
TARGET_PERCENTILES = [0.1, 0.5, 0.9]  # low, medium, high degree start nodes
N_BFS_SETS = 5  # per percentile
BFS_MODULE_SIZE = 50  # -> 40 seeds, 10 hidden
MAX_OVERLAP = 0.3  # Jaccard between two modules of the same network

# community
COMMUNITY_ALGORITHM = "leiden"
COMMUNITY_SEED = SEEDS[0]  # first seed of the community pipeline
N_COMMUNITY_SETS = 10
MIN_COMMUNITY_SIZE = 20  # at least 4 hidden nodes
MAX_COMMUNITY_SIZE = 150  # larger communities are split (at most 120 seeds)


def seed_sets_of_unit(unit, units):
    baselines, perturbed_folder, _ = units[unit]
    network, model = os.path.normpath(perturbed_folder).split(os.sep)[-2:]
    node_to_idx = load_node_index(units[network][0][0][0])
    labels = np.array(list(node_to_idx))
    rows = []
    for replicate, (edgelist, _) in enumerate(baselines):
        edges = read_baseline_edges(edgelist, node_to_idx)
        adj = edges_to_adjacency(edges, len(node_to_idx))
        # own reproducible random stream per network
        rng = np.random.default_rng(zlib.crc32(f"{network}|{model}|{replicate}".encode()))
        print(f"{network}/{model}/{replicate}:", flush=True)

        records = [
            {"seed_type": "bfs", **r}
            for r in bfs_seed_sets(
                adj, TARGET_PERCENTILES, N_BFS_SETS, BFS_MODULE_SIZE,
                SEED_FRACTION, MAX_OVERLAP, rng,
            )
        ]
        partition = detect_partitions(
            edges, len(node_to_idx), COMMUNITY_ALGORITHM, seeds=(COMMUNITY_SEED,)
        )[0]
        partition, parents, depths = split_large_communities(
            adj, partition, MAX_COMMUNITY_SIZE, seed=COMMUNITY_SEED
        )
        records += [
            {"seed_type": "community", **r}
            for r in community_seed_sets(
                partition, N_COMMUNITY_SETS, MIN_COMMUNITY_SIZE, MAX_COMMUNITY_SIZE,
                SEED_FRACTION, rng, parents, depths,
            )
        ]

        for set_id, record in enumerate(records):
            seeds, hidden = record.pop("seeds"), record.pop("hidden")
            if "start_node" in record:
                record["start_node"] = labels[record["start_node"]]
            rows.append(
                {
                    "network": network,
                    "model": model,
                    "replicate": replicate,
                    "set_id": set_id,
                    **record,
                    "n_seeds": len(seeds),
                    "n_hidden": len(hidden),
                    "seed_nodes": ";".join(labels[seeds]),
                    "hidden_nodes": ";".join(labels[hidden]),
                }
            )
    return rows


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    units = all_units()
    with ProcessPoolExecutor(n_workers) as pool:
        results = pool.map(seed_sets_of_unit, units, [units] * len(units))
        table = pd.DataFrame([row for rows in results for row in rows])

    columns = [
        "network", "model", "replicate", "set_id", "seed_type", "n_seeds", "n_hidden",
        "target_percentile", "target_degree", "start_node", "start_degree", "max_overlap",
        "community_id", "community_size", "parent_community_id", "split_depth",
        "seed_nodes", "hidden_nodes",
    ]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    table[columns].to_csv(OUT_PATH, index=False)

    counts = table.groupby(["network", "model", "replicate", "seed_type"]).size().unstack(fill_value=0)
    expected = {"bfs": N_BFS_SETS * len(TARGET_PERCENTILES), "community": N_COMMUNITY_SETS}
    print(f"{len(table)} seed sets for {len(counts)} networks -> {OUT_PATH}")
    for seed_type, n in expected.items():
        short = counts[counts[seed_type] < n][seed_type]
        if len(short):
            print(f"fewer than {n} {seed_type} sets:\n{short.to_string()}")
