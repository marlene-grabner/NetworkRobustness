"""
Common interface for all seed-expansion algorithms:

    score(adj: scipy.sparse.csr_matrix, seed_idx: np.ndarray, **params) -> np.ndarray

`adj` is a symmetric CSR adjacency matrix over the FULL node index (same
dimension for baseline and every perturbed repeat of that network).
`seed_idx` are matrix indices (not raw node ids) into that same space.
Returns a dense float score vector of length n_nodes, higher = more relevant.
Seed nodes themselves get score -inf so they never appear as "discovered" hits.

Register new algorithms in ALGORITHMS so notebooks/run_*.py can look them up
by name without importing each module individually.
"""
from .rwr import rwr_row_normalized, rwr_symmetric_normalized
from .diamond import diamond_score
from .first_neighbors import first_neighbors_score

ALGORITHMS = {
    "rwr_row": rwr_row_normalized,
    "rwr_sym": rwr_symmetric_normalized,
    "diamond": diamond_score,
    "first_neighbors": first_neighbors_score,
}