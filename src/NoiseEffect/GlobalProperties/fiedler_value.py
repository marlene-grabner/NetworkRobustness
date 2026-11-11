"""Algebraic connectivity (Fiedler value) of the largest connected component."""

import warnings

import numpy as np
from scipy.sparse import diags
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh, lobpcg

SMALL_GRAPH = 50  # below this the Laplacian is solved densely
MAX_RESIDUAL = 1e-3  # accepted |Lx - lambda x| / lambda of the fast solver


def fiedler_on_gcc(A, N):
    """
    Returns the Fiedler value (second-smallest eigenvalue of the unnormalised
    Laplacian) of the largest connected component, and whether the graph was
    disconnected.

    LOBPCG is fast on the dense networks but does not converge on sparse,
    grid-like ones (power grid, roads), where it returns a too large value
    without an error. Its result is therefore only kept if the residual is
    small; otherwise the exact shift-invert solver is used, which is fast on
    exactly those sparse networks.
    """
    n_comp, labels = connected_components(A, directed=False, return_labels=True)
    disconnected = n_comp > 1
    if disconnected:
        mask = labels == np.bincount(labels).argmax()
        A = A.tocsr()[mask][:, mask]
        N = int(mask.sum())

    L = laplacian(A, normed=False).astype(float).tocsr()
    if N < SMALL_GRAPH:
        if N < 2:
            return 0.0, disconnected
        return max(0.0, float(np.linalg.eigvalsh(L.toarray())[1])), disconnected

    fiedler = _fiedler_lobpcg(L)
    if fiedler is None:
        # L + 1e-6 I is positive definite, so the factorisation is stable
        evals = eigsh(
            L.tocsc(), k=2, sigma=-1e-6, which="LM", return_eigenvectors=False
        )
        fiedler = float(np.sort(evals)[1])
    return max(0.0, fiedler), disconnected


def _fiedler_lobpcg(L) -> float | None:
    """Fiedler value with LOBPCG, or None if it did not converge."""
    M = diags(
        1.0 / L.diagonal()
    )  # Jacobi preconditioner (no zero degrees in a component)
    X = np.random.default_rng(42).standard_normal((L.shape[0], 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            evals, evecs = lobpcg(L, X, M=M, largest=False, tol=1e-7, maxiter=500)
        except Exception:
            return None
    i = np.argsort(evals)[1]
    value, x = float(evals[i]), evecs[:, i] / np.linalg.norm(evecs[:, i])
    if not np.isfinite(value) or value <= 0:
        return None
    if np.linalg.norm(L @ x - value * x) > MAX_RESIDUAL * value:
        return None
    return value
