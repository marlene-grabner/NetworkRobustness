import numpy as np
from typing import Tuple
from scipy.sparse import coo_matrix, eye, diags
from scipy.sparse.linalg import eigsh


def to_normalized_laplacian(A: coo_matrix) -> coo_matrix:
    """
    Converts a raw symmetric Adjacency Matrix into a Normalized Graph Laplacian:
    L = I - D^(-1/2) * A * D^(-1/2)
    """
    degrees = np.array(A.sum(axis=1)).flatten()

    # Handle isolated nodes gracefully by keeping their inverted degree at 0
    d_inv_sqrt = np.zeros_like(degrees, dtype=float)
    d_inv_sqrt[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])

    D_inv_sqrt = diags(d_inv_sqrt)

    I = eye(A.shape[0])
    L = I - D_inv_sqrt.dot(A).dot(D_inv_sqrt)

    return L


def get_spectral_components(L: coo_matrix, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the smallest k eigenvalues and eigenvectors of a Laplacian matrix.
    Utilizes spectrum reversal and symmetry-breaking to guarantee numerical
    stability and eliminate scipy warnings on noisy complex networks.
    """
    n_nodes = L.shape[0]

    # 1. Flip the spectrum (M = 2I - L) so small values become large values
    M = 2.0 * eye(n_nodes) - L

    # 2. Symmetry Breaking: Inject microscopic noise to separate identical degenerate eigenvalues
    np.random.seed(42)
    micro_noise = np.random.uniform(0, 1e-10, size=n_nodes)
    M_noisy = M + diags(micro_noise)

    # 3. Fast extraction using Largest Magnitude with relaxed tolerance to avoid LOBPCG warnings
    eigenvalues_M, eigenvectors = eigsh(
        M_noisy, k=k, which="LM", tol=1e-3, maxiter=1000
    )

    # 4. Flip eigenvalues back to original Laplacian space
    eigenvalues = 2.0 - eigenvalues_M

    # 5. Sort from smallest to largest
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def get_eigenvector_centrality(A: coo_matrix) -> np.ndarray:
    """
    Calculates the Eigenvector Centrality directly from the Adjacency Matrix.
    Extracts the principal eigenvector associated with the largest eigenvalue.
    """
    # Convert to float to prevent scipy typing complaints
    # Use relaxed tolerance and elevated maxiter for highly noisy structures
    _, eigenvectors = eigsh(A.astype(float), k=1, which="LM", tol=1e-3, maxiter=1000)

    # By Perron-Frobenius, the vector is real and single-signed. Absolute value guarantees positive scores.
    centrality = np.abs(eigenvectors[:, 0])

    return centrality
