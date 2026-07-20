"""
First-neighbors baseline: rank every non-seed node by how many edges it has
into the seed set. Ties are broken implicitly by argsort stability; if you
want an explicit tie-break by degree, see the commented line below.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def first_neighbors_score(adj: sparse.csr_matrix, seed_idx: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    seed_vec = np.zeros(n, dtype=np.float64)
    seed_vec[seed_idx] = 1.0

    scores = adj @ seed_vec  # scores[i] = number of edges from i into the seed set

    # degree tie-break (nudge ties toward lower-degree nodes):
    degree = np.asarray(adj.sum(axis=1)).ravel()
    scores = scores - 1e-6 * degree

    scores[seed_idx] = -np.inf
    return scores
