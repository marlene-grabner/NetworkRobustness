"""
Random Walk with Restart (RWR), implemented as sparse matrix-vector power
iteration -- O(nnz) per iteration, no networkx, no dense matrices.

Two normalizations of the adjacency matrix A (degree vector d = A.sum(axis=1)):

  row  : W = D^-1 A            (standard transition matrix; each row sums to 1)
         propagation uses W^T = A D^-1, i.e. p' = A @ (p / d)
  sym  : W = D^-1/2 A D^-1/2   (symmetric; used e.g. in label propagation)
         propagation uses W directly (W is symmetric), i.e.
         p' = d^-1/2 * (A @ (p * d^-1/2))

Both update rules:
    p_{t+1} = (1 - restart) * W_prop(p_t) + restart * p0
iterated to convergence (L1 change < tol) or max_iter.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _restart_vector(n_nodes: int, seed_idx: np.ndarray) -> np.ndarray:
    p0 = np.zeros(n_nodes, dtype=np.float64)
    p0[seed_idx] = 1.0 / len(seed_idx)
    return p0


def _degree(adj: sparse.csr_matrix) -> np.ndarray:
    deg = np.asarray(adj.sum(axis=1)).ravel()
    return deg


def rwr_row_normalized(adj: sparse.csr_matrix, seed_idx: np.ndarray,
                        restart: float = 0.7, tol: float = 1e-8,
                        max_iter: int = 1000) -> np.ndarray:
    n = adj.shape[0]
    p0 = _restart_vector(n, seed_idx)
    deg = _degree(adj)
    deg_inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)

    p = p0.copy()
    for _ in range(max_iter):
        p_next = (1 - restart) * (adj @ (p * deg_inv)) + restart * p0
        if np.abs(p_next - p).sum() < tol:
            p = p_next
            break
        p = p_next

    p[seed_idx] = -np.inf
    return p


def rwr_symmetric_normalized(adj: sparse.csr_matrix, seed_idx: np.ndarray,
                              restart: float = 0.7, tol: float = 1e-8,
                              max_iter: int = 1000) -> np.ndarray:
    n = adj.shape[0]
    p0 = _restart_vector(n, seed_idx)
    deg = _degree(adj)
    deg_inv_sqrt = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)

    p = p0.copy()
    for _ in range(max_iter):
        p_next = (1 - restart) * (deg_inv_sqrt * (adj @ (p * deg_inv_sqrt))) + restart * p0
        if np.abs(p_next - p).sum() < tol:
            p = p_next
            break
        p = p_next

    p[seed_idx] = -np.inf
    return p