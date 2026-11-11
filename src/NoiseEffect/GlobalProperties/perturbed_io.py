"""
Reading the perturbed network files written by NoiseNetworks:

  .../{network}/{model}/{noise_type}_{level}.parquet
  columns: source, target, replicate, repeat

One file holds many networks: one per (replicate, repeat). Replicates are the
draws of a null model (always 0 for an empirical network), repeats the
perturbations of one replicate.
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ACTIONS = ("addition", "removal", "rewiring")


def parse_perturbed_path(path: str) -> dict:
    """
    Network, model, noise type and level from the file path, e.g.
    .../huri/er/hub_addition_0p1.parquet -> network huri, model er,
    noise_type hub_addition, target hub, action addition, level 0.1
    """
    folder, filename = os.path.split(os.path.abspath(path))
    noise_type, level = filename.removesuffix(".parquet").rsplit("_", 1)
    target, action = noise_type.rsplit("_", 1)
    if action not in ACTIONS:
        raise ValueError(f"Can not read the noise type from {filename}")
    return {
        "network": os.path.basename(os.path.dirname(folder)),
        "model": os.path.basename(folder),
        "noise_type": noise_type,
        "target": target,
        "action": action,
        "level": float(level.replace("p", ".")),
    }


def load_node_index(path_to_edgelist: str, path_to_nodelist: str | None = None) -> dict:
    """
    Node label -> integer 0..n-1 for a baseline edgelist (headerless csv).
    The optional nodelist holds degree-0 nodes missing from the edgelist.

    All null model replicates of a network have the node set of the empirical
    network, so its index can be used for all models of that network.
    """
    edges = pd.read_csv(path_to_edgelist, header=None, dtype=str)
    nodes = set(edges[0]) | set(edges[1])
    if path_to_nodelist is not None:
        with open(path_to_nodelist) as f:
            nodes |= {n.strip() for n in f.read().split(",") if n.strip()}
    return {node: i for i, node in enumerate(sorted(nodes))}


def read_baseline_edges(path_to_edgelist: str, node_to_idx: dict) -> np.ndarray:
    """Baseline edgelist as an (m, 2) integer array, like the perturbed networks."""
    edges = pd.read_csv(path_to_edgelist, header=None, dtype=str)
    return np.column_stack([edges[0].map(node_to_idx), edges[1].map(node_to_idx)])


def iter_perturbed_networks(path: str, node_to_idx: dict):
    """
    Yields (replicate, repeat, edges) for every network in a file, with edges
    as an (m, 2) integer array. The file is read one row group at a time, so
    only a part of it is in memory.
    """
    labels = pa.array(list(node_to_idx), type=pa.string())
    file = pq.ParquetFile(path)
    current, parts = None, []
    for i in range(file.num_row_groups):
        table = file.read_row_group(i)
        edges = np.column_stack(
            [_to_index(table[c], labels, path) for c in ("source", "target")]
        )
        # a 16 bit key for the network: replicate and repeat are both uint8
        keys = (
            table["replicate"].to_numpy().astype(np.uint16) * 256
            + table["repeat"].to_numpy()
        )
        # rows of a network are contiguous, but it can span row groups
        bounds = np.flatnonzero(np.diff(keys)) + 1
        for key, block in zip(keys[np.r_[0, bounds]], np.split(edges, bounds)):
            if key != current:
                if parts:
                    yield *divmod(int(current), 256), np.concatenate(parts)
                current, parts = key, []
            parts.append(block)
    if parts:
        yield *divmod(int(current), 256), np.concatenate(parts)


def _to_index(column, labels, path) -> np.ndarray:
    index = pc.index_in(column.cast(pa.string()), value_set=labels)
    if index.null_count:
        raise ValueError(f"{path} has nodes that are not in the baseline network.")
    return index.to_numpy()
