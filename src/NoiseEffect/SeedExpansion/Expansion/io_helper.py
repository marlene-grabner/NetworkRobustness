"""
I/O and graph-construction helpers.

Design decision: the node<->index mapping is built ONCE from each baseline
network's edgelist and then reused for every perturbed variant of that same
network. This keeps seed indices, adjacency dimensions and score vectors
aligned across baseline/perturbed comparisons without re-mapping per repeat.

If a perturbed edgelist references a node that isn't in the baseline mapping
(e.g. an "addition" perturbation that introduces new nodes), those edges are
dropped by default and a count is returned so you can sanity-check how often
this happens. Set `extend_index=True` in `edges_to_sparse` if you'd rather
keep such nodes (they'll just get near-zero RWR scores and be ignored by
metrics computed against the baseline node set).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy import sparse


# --------------------------------------------------------------------------- #
# Node index
# --------------------------------------------------------------------------- #

@dataclass
class NodeIndex:
    network: str
    nodes: np.ndarray          # nodes[i] = original node id, position i = matrix index
    node_to_idx: dict          # node id -> matrix index

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def to_idx(self, node_ids) -> np.ndarray:
        """Map an array-like of node ids to matrix indices. Unknown ids -> -1."""
        if len(node_ids) == 0:
            return np.array([], dtype=np.int64)
            
        # 1. Detect the expected type from our dictionary keys
        expected_type = type(next(iter(self.node_to_idx.keys())))
        
        # 2. Convert incoming IDs to match that type safely
        try:
            casted_ids = np.asarray(node_ids, dtype=expected_type)
        except (ValueError, TypeError):
            # Fallback to a element-by-element cast if vector casting fails
            casted_ids = [expected_type(n) for n in node_ids]
            
        # 3. Perform the dictionary lookup
        return np.array([self.node_to_idx.get(n, -1) for n in casted_ids], dtype=np.int64)


def build_node_index(network: str, edge_df: pd.DataFrame,
                      source_col: str = "source", target_col: str = "target") -> NodeIndex:
    nodes = pd.unique(pd.concat([edge_df[source_col], edge_df[target_col]], ignore_index=True))
    nodes = np.sort(nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    return NodeIndex(network=network, nodes=nodes, node_to_idx=node_to_idx)


def save_node_index(index: NodeIndex, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_node_index(path: str | Path) -> NodeIndex:
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# Adjacency construction
# --------------------------------------------------------------------------- #

def edges_to_sparse(edge_df: pd.DataFrame, index: NodeIndex,
                     source_col: str = "source", target_col: str = "target",
                     weight_col: str | None = None,
                     extend_index: bool = False) -> tuple[sparse.csr_matrix, int]:
    """
    Build a symmetric (undirected) CSR adjacency matrix sized (n_nodes, n_nodes)
    from an edgelist dataframe, using `index` for node->matrix-index mapping.

    Returns (adjacency, n_dropped_edges) where n_dropped_edges counts edges
    touching a node id not present in `index` (only relevant if
    extend_index=False, which is the default).
    """
    src = index.to_idx(edge_df[source_col].values)
    tgt = index.to_idx(edge_df[target_col].values)

    if extend_index:
        raise NotImplementedError(
            "extend_index=True requires rebuilding NodeIndex per-repeat; "
            "not implemented because it breaks index alignment across repeats. "
            "Filter/relabel new nodes upstream if you need them."
        )

    valid = (src >= 0) & (tgt >= 0)
    n_dropped = int((~valid).sum())
    src, tgt = src[valid], tgt[valid]

    if weight_col is not None:
        w = edge_df[weight_col].values[valid].astype(np.float64)
    else:
        w = np.ones(len(src), dtype=np.float64)

    n = index.n_nodes
    # symmetrize by adding both directions
    rows = np.concatenate([src, tgt])
    cols = np.concatenate([tgt, src])
    data = np.concatenate([w, w])

    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    adj.sum_duplicates()
    if weight_col is None:
        # sum_duplicates() adds up repeated unweighted edges; re-binarize so
        # duplicate/multi-edges don't silently create weighted hubs
        adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj, n_dropped


def load_baseline_edgelist(path: str | Path, source_col: str = "source",
                            target_col: str = "target") -> pd.DataFrame:
    return pd.read_csv(path, names = [source_col, target_col])


def iter_perturbed_repeats(parquet_path: str | Path, index: NodeIndex,
                            source_col: str = "source", target_col: str = "target",
                            repeat_col: str = "repeat") -> Iterator[tuple[int, sparse.csr_matrix, int]]:
    """
    Stream one repeat at a time from a perturbed-network parquet file, yielding
    (repeat_id, adjacency, n_dropped_edges).

    Reads the whole parquet once (they're one file per noise level, not per
    repeat, so this is a single I/O call) then groups in-memory -- far cheaper
    than re-opening the file per repeat.
    """
    import pyarrow.parquet as pq
    df = pq.read_table(parquet_path).to_pandas()
    for repeat_id, sub in df.groupby(repeat_col, sort=True):
        adj, n_dropped = edges_to_sparse(sub, index, source_col=source_col, target_col=target_col)
        yield int(repeat_id), adj, n_dropped


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #

def load_seed_table(seeds_csv: str | Path) -> pd.DataFrame:
    """
    Loads the seed table and preserves the distinction between separate seed IDs.
    """
    df = pd.read_csv(seeds_csv)
    if "seed_nodes" not in df.columns or "seed_id" not in df.columns:
        raise ValueError("seeds_csv must have 'seed_id' and 'seed_nodes' columns")
    return df

def iter_network_seeds(seed_table: pd.DataFrame, network: str, index: NodeIndex) -> Iterator[tuple[str, np.ndarray]]:
    """
    Yields pairs of (seed_id, seed_indices_ndarray) for a specific network.
    Filters out missing nodes and ensures matrix index alignment.
    """
    network_df = seed_table[seed_table["network_id"] == network]
    
    for _, row in network_df.iterrows():
        seed_id = row["seed_id"]
        # Handle cases where seed_nodes might be empty or malformed
        if pd.isna(row["seed_nodes"]):
            continue
            
        node_ids = str(row["seed_nodes"]).split(';')
        
        # Cast to match the NodeIndex type if possible
        try:
            node_ids = np.array(node_ids, dtype=index.nodes.dtype)
        except (ValueError, TypeError):
            node_ids = np.array(node_ids)
            
        idx = index.to_idx(node_ids)
        valid_idx = idx[idx >= 0]
        
        if len(valid_idx) == 0:
            print(f"[io_helper] Warning: All seeds for {seed_id} were dropped!")
            continue
            
        yield seed_id, valid_idx


def get_seed_indices(seed_table: pd.DataFrame, network: str, seed_id: str, index: NodeIndex) -> np.ndarray:
    """
    Extracts the matrix node indices for a SPECIFIC seed_id in a given network.
    """
    # Filter for the specific network AND the specific seed_id configuration
    mask = (seed_table["network_id"] == network) & (seed_table["seed_id"] == seed_id)
    row = seed_table[mask]
    
    if row.empty:
        print(f"[io_helper] warning: No entry found for network '{network}' with seed_id '{seed_id}'")
        return np.array([], dtype=np.int64)
        
    # Grab the semicolon-separated string from 'seed_nodes'
    seed_string = row["seed_nodes"].values[0]
    if pd.isna(seed_string):
        return np.array([], dtype=np.int64)
        
    # Split string into individual node IDs
    ids = np.array(str(seed_string).split(";"))
    
    # Cast to the same dtype as the node index where possible
    try:
        ids = ids.astype(index.nodes.dtype)
    except (ValueError, TypeError):
        pass
        
    idx = index.to_idx(ids)
    valid_idx = idx[idx >= 0]
    
    missing = len(ids) - len(valid_idx)
    if missing > 0:
        print(f"[io_helper] warning: {missing}/{len(ids)} nodes for seed '{seed_id}' "
              f"not found in node index")
              
    return valid_idx


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

def save_ranking(scores: np.ndarray, index: NodeIndex, path: str | Path,
                  extra_cols: dict | None = None) -> None:
    """Save a full ranking (baseline case) as parquet: node, node_idx, score, rank."""
    order = np.argsort(-scores, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    out = pd.DataFrame({
        "node": index.nodes,
        "node_idx": np.arange(index.n_nodes),
        "score": scores,
        "rank": rank,
    })
    if extra_cols:
        for k, v in extra_cols.items():
            out[k] = v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def load_ranking(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def append_metrics_rows(rows: list[dict], path: str | Path) -> None:
    """Write accumulated metric rows for one SLURM task to a single parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)