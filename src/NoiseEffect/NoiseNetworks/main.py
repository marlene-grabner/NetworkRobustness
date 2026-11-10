"""
Generates perturbed networks from a baseline network by adding, removing or
rewiring edges. Each noise type lives in its own module:

  random_noise.py      random addition and removal
  targeted_noise.py    hub-, periphery- and betweenness-weighted noise
  structural_noise.py  triadic closure (RA type) and degree-preserving rewiring
  baseline_io.py       loading the baseline networks

Output: one parquet file per (noise type, noise level) in the output folder,
named {noise_type}_{level}.parquet (e.g. hub_addition_0p1.parquet),
with the columns source, target, replicate, repeat. A file holds every
perturbed network of that type and level: for an empirical network one
replicate, for a null model one replicate per null model draw.
"""

import os
import random
import zlib
from functools import partial

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .baseline_io import _loadBaseline
from .random_noise import _randomEdgesToAdd, _randomEdgesToRemove
from .structural_noise import (
    _countTriadicCandidates,
    _rewireEdges,
    _triadicClosureEdgesToAdd,
)
from .targeted_noise import _edgeBetweenness, _targetedAddition, _targetedRemoval

# noise type -> function(g, num_modify, graph_info=...) returning the edges to add
# (graph_info is passed by keyword, because partial() fixes 'target' before it)
ADDITIONS = {
    "random_addition": _randomEdgesToAdd,
    "hub_addition": partial(_targetedAddition, target="hubs"),
    "periphery_addition": partial(_targetedAddition, target="periphery"),
    "triadic_closure_addition": _triadicClosureEdgesToAdd,
}
# noise type -> function(g, num_modify, graph_info=...) returning the edges to remove
REMOVALS = {
    "random_removal": _randomEdgesToRemove,
    "hub_removal": partial(_targetedRemoval, target="hubs"),
    "periphery_removal": partial(_targetedRemoval, target="periphery"),
    "betweenness_removal": partial(_targetedRemoval, target="betweenness"),
}
# noise type -> function(g, num_modify, graph_info=...) returning the perturbed graph
REWIRINGS = {
    "degree_preserving_rewiring": _rewireEdges,
}
ALL_NOISE_TYPES = [*ADDITIONS, *REMOVALS, *REWIRINGS]

SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("target", pa.string()),
        ("replicate", pa.uint8()),
        ("repeat", pa.uint8()),
    ]
)


def generatePerturbedNetworks(
    baselines: list[tuple[str, str | None]],
    folder_to_save_perturbed: str,
    noise_levels: dict[str, list[float]],
    num_repeats: int,
    seed: int = 0,
) -> list[tuple[str, float]]:
    """
    Perturbs every baseline num_repeats times per noise type and level.

    :param baselines: one (edgelist csv, isolated-nodes file or None) per
        replicate: a single entry for an empirical network, one per draw for a
        null model. The replicate index is the position in this list.
    :param folder_to_save_perturbed: output folder (existing files are never
        overwritten)
    :param noise_levels: noise type -> fractions of the baseline edges to add,
        remove or rewire
    :param num_repeats: perturbed networks per replicate, type and level
    :param seed: base seed; every (replicate, type, level) gets its own seed
        derived from it, so a file can be regenerated on its own
    :return: the (noise type, level) pairs skipped because the level is not
        reachable (triadic closure with too few candidate pairs)
    """
    unknown = set(noise_levels) - set(ALL_NOISE_TYPES)
    if unknown:
        raise ValueError(f"Unknown noise types: {sorted(unknown)}")
    tasks = [(t, level) for t, levels in noise_levels.items() for level in levels]

    os.makedirs(folder_to_save_perturbed, exist_ok=True)
    paths = {
        task: os.path.join(folder_to_save_perturbed, _fileName(*task)) for task in tasks
    }
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing:
        raise FileExistsError(
            f"{len(existing)} output files exist already, e.g. {existing[0]}"
        )

    skipped = _unreachableLevels(baselines, tasks)
    for noise_type, level in skipped:
        print(f"  skipping {noise_type} at {level}: more edges than candidate pairs")
    tasks = [task for task in tasks if task not in skipped]

    writers = {}
    try:
        for replicate, (edgelist, nodelist) in enumerate(baselines):
            g, idx_to_node, _ = _loadBaseline(edgelist, nodelist)
            graph_info = _precomputeGraphInfo(g, [t for t, _ in tasks])
            for noise_type, level in tasks:
                _seed(seed, replicate, noise_type, level)
                num_modify = _numberOfEdgesToModify(noise_type, level, graph_info)
                table = _perturbRepeats(
                    g,
                    num_modify,
                    noise_type,
                    graph_info,
                    idx_to_node,
                    replicate,
                    num_repeats,
                )
                if (noise_type, level) not in writers:
                    writers[(noise_type, level)] = pq.ParquetWriter(
                        paths[(noise_type, level)], SCHEMA
                    )
                # one row group per replicate
                writers[(noise_type, level)].write_table(table)
            print(f"  replicate {replicate} done", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    return skipped


def _fileName(noise_type: str, level: float) -> str:
    return f"{noise_type}_{str(level).replace('.', 'p')}.parquet"


def _seed(seed: int, replicate: int, noise_type: str, level: float):
    """Seeds random and numpy for one (replicate, type, level)."""
    task_seed = zlib.crc32(f"{seed}|{replicate}|{noise_type}|{level}".encode())
    random.seed(task_seed)
    np.random.seed(task_seed)


def _unreachableLevels(baselines, tasks) -> list[tuple[str, float]]:
    """
    Triadic closure levels that need more edges than some replicate has
    candidate pairs. They are skipped for all replicates, so every file holds
    the same replicates.
    """
    triadic = [task for task in tasks if task[0] == "triadic_closure_addition"]
    if not triadic:
        return []
    unreachable = set()
    for edgelist, nodelist in baselines:
        g, _, _ = _loadBaseline(edgelist, nodelist)
        n_candidates = _countTriadicCandidates(g)
        for noise_type, level in triadic:
            if int(g.number_of_edges() * level) > n_candidates:
                unreachable.add((noise_type, level))
    return [task for task in triadic if task in unreachable]


def _precomputeGraphInfo(g: nx.Graph, noise_types: list[str]) -> dict:
    """Baseline information the noise types sample from (computed once)."""
    graph_info = {
        "degrees": dict(g.degree()),
        "nodes": list(g.nodes()),
        "edges": list(g.edges()),
    }
    # The expensive parts are only computed when a noise type needs them
    if "betweenness_removal" in noise_types:
        graph_info["edge_betweenness"] = _edgeBetweenness(g, graph_info["edges"])
    if "triadic_closure_addition" in noise_types:
        graph_info["neighbours"] = {n: list(g.neighbors(n)) for n in g.nodes()}
        graph_info["n_triadic_candidates"] = _countTriadicCandidates(g)
    return graph_info


def _numberOfEdgesToModify(noise_type: str, level: float, graph_info: dict) -> int:
    """Number of edges to add, remove or rewire for a noise level."""
    num_edges = len(graph_info["edges"])
    num_modify = int(num_edges * level)
    if num_modify <= 0:
        raise ValueError(f"Noise level {level} modifies no edges.")
    if noise_type in REMOVALS and num_modify >= num_edges:
        raise ValueError("Can not remove all edges in the graph.")
    if noise_type in REWIRINGS and num_modify > num_edges:
        raise ValueError("Can not rewire more edges than the graph has.")
    return num_modify


def _perturb(
    g: nx.Graph, num_modify: int, noise_type: str, graph_info: dict
) -> nx.Graph:
    """Returns one perturbed copy of the baseline."""
    if noise_type in ADDITIONS:
        G_perturbed = g.copy()
        G_perturbed.add_edges_from(
            ADDITIONS[noise_type](g, num_modify, graph_info=graph_info)
        )
    elif noise_type in REMOVALS:
        G_perturbed = g.copy()
        G_perturbed.remove_edges_from(
            REMOVALS[noise_type](g, num_modify, graph_info=graph_info)
        )
    else:
        G_perturbed = REWIRINGS[noise_type](g, num_modify, graph_info=graph_info)
    return G_perturbed


def _perturbRepeats(
    g, num_modify, noise_type, graph_info, idx_to_node, replicate, num_repeats
):
    """All repeats of one (replicate, type, level) as one arrow table."""
    labels = np.array([idx_to_node[i] for i in range(len(idx_to_node))], dtype=object)
    edge_arrays, repeats = [], []
    for repeat in range(num_repeats):
        G_perturbed = _perturb(g, num_modify, noise_type, graph_info)
        edges = np.array(G_perturbed.edges(), dtype=np.int64).reshape(-1, 2)
        edge_arrays.append(edges)
        repeats.append(np.full(len(edges), repeat, dtype=np.uint8))
    edges = np.concatenate(edge_arrays)
    return pa.table(
        {
            "source": labels[edges[:, 0]],  # back to the original node labels
            "target": labels[edges[:, 1]],
            "replicate": np.full(len(edges), replicate, dtype=np.uint8),
            "repeat": np.concatenate(repeats),
        },
        schema=SCHEMA,
    )
