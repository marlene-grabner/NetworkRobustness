"""
Thin orchestration layer over src/algorithms. Keeps notebooks/*.py scripts
free of algorithm-specific logic -- they just call `run_algorithm`.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .Algorithms import ALGORITHMS

# Per-algorithm default parameters. Override via `params` in run_algorithm.
DEFAULT_PARAMS = {
    "rwr_row": dict(restart=0.7, tol=1e-8, max_iter=200),
    "rwr_sym": dict(restart=0.7, tol=1e-8, max_iter=200),
    "diamond": dict(max_added_nodes=200, alpha=1),
    "first_neighbors": dict(),
}


def run_algorithm(name: str, adj: sparse.csr_matrix, seed_idx: np.ndarray,
                   params: dict | None = None) -> np.ndarray:
    if name not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {list(ALGORITHMS)}")
    if len(seed_idx) == 0:
        raise ValueError("seed_idx is empty -- check seed table / node index alignment")

    p = dict(DEFAULT_PARAMS.get(name, {}))
    if params:
        p.update(params)
    return ALGORITHMS[name](adj, seed_idx, **p)


def run_all_algorithms(adj: sparse.csr_matrix, seed_idx: np.ndarray,
                        algorithms: list[str] | None = None,
                        params: dict[str, dict] | None = None) -> dict[str, np.ndarray]:
    algorithms = algorithms or list(ALGORITHMS)
    params = params or {}
    return {name: run_algorithm(name, adj, seed_idx, params.get(name)) for name in algorithms}