import os, sys, time, logging
import numpy as np
import pandas as pd
import warnings
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import lobpcg, eigsh

def fiedler_on_gcc(A, N):
    """Returns Fiedler value on GCC, plus whether graph was disconnected."""
    n_comp, labels = connected_components(A, directed=False, return_labels=True)
    
    disconnected = n_comp > 1
    if disconnected:
        # Slice out the GCC
        gcc_label = np.bincount(labels).argmax()
        mask = labels == gcc_label
        A = A.tocsr()[mask][:, mask]
        N = mask.sum()

    L = laplacian(A, normed=False).asfptype().tocsr()
    diag = L.diagonal().copy()
    diag[diag == 0] = 1.0
    M = diags(1.0 / diag)

    rng = np.random.default_rng(42)
    X = rng.standard_normal((N, 2))

    fiedler = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            evals, _ = lobpcg(L, X, M=M, largest=False, tol=1e-5, maxiter=500)
            candidate = float(np.sort(evals)[1])
            # accept only if it's a sane Fiedler value
            if np.isfinite(candidate) and candidate >= -1e-8:
                fiedler = max(0.0, candidate)
        except Exception:
            fiedler = None  # failure -> fall through to shift-invert

    # fallback only if LOBPCG threw or returned something invalid
    if fiedler is None:
        evals = eigsh(L, k=2, sigma=1e-8, which='LM', return_eigenvectors=False)
        fiedler = max(0.0, float(np.sort(evals)[1]))

    return fiedler, disconnected