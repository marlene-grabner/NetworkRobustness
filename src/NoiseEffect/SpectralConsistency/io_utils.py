import os
import re
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


def parse_filename(filepath: str) -> dict:
    """
    Extracts metadata from the perturbed network filename.
    Expected format example: 'chloe_ppi_added_edges_noise0p1_repeat2.txt'
    """
    filename = os.path.basename(filepath)
    metadata = {
        "Filename": filename,
        "Noise_Type": "unknown",
        "Noise_Level": np.nan,
        "Repeat_ID": np.nan,
    }

    # 1. Determine Noise Type
    if "added" in filename.lower():
        metadata["Noise_Type"] = "added"
    elif "removed" in filename.lower() or "deletion" in filename.lower():
        metadata["Noise_Type"] = "removed"

    # 2. Extract Noise Level (looks for 'noise' followed by numbers, handling 'p' as decimal)
    # Example match: 'noise0p1' or 'noise1p5' or 'noise50'
    noise_match = re.search(r"noise(\d+p\d+|\d+)", filename, re.IGNORECASE)
    if noise_match:
        noise_str = noise_match.group(1).replace("p", ".")
        metadata["Noise_Level"] = float(noise_str)

    # 3. Extract Repeat ID (looks for 'repeat' followed by digits)
    # Example match: 'repeat2'
    repeat_match = re.search(r"repeat(\d+)", filename, re.IGNORECASE)
    if repeat_match:
        metadata["Repeat_ID"] = int(repeat_match.group(1))

    return metadata


def get_universal_node_mapping(orig_file: str, pert_file: str) -> dict:
    """
    Reads both edgelists to find all unique nodes and creates a universal
    mapping from Node ID to a consistent integer index (0 to N-1).
    Ensures matrix dimensions match for downstream math.
    """
    df_orig = pd.read_csv(orig_file, sep="\t", header=None, names=["source", "target"])
    df_pert = pd.read_csv(pert_file, sep="\t", header=None, names=["source", "target"])

    all_nodes = set(df_orig["source"]).union(set(df_orig["target"]))
    all_nodes = all_nodes.union(set(df_pert["source"])).union(set(df_pert["target"]))

    node_to_idx = {node: i for i, node in enumerate(sorted(all_nodes))}
    return node_to_idx


def load_adjacency(filepath: str, node_to_idx: dict) -> coo_matrix:
    """
    Loads a TSV or TXT edgelist and constructs a symmetric, binarized
    raw Adjacency Matrix using a predefined node mapping.
    """
    df = pd.read_csv(filepath, sep="\t", header=None, names=["source", "target"])

    # Map string names to integer indices
    row = df["source"].map(node_to_idx).values
    col = df["target"].map(node_to_idx).values
    data = np.ones(len(row))

    n_nodes = len(node_to_idx)

    # Construct Sparse Matrix and ensure undirected symmetry
    A = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    A = A + A.T

    # Binarize to cap any duplicate edges at 1
    A.data = np.where(A.data > 0, 1, 0)

    return A
