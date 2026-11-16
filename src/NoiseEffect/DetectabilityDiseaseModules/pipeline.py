import networkx as nx
import pandas as pd
import numpy as np
from scipy.sparse.csgraph import shortest_path
import re
import sys
import os
import importlib.util
from pathlib import Path
from tqdm import tqdm
from .evaluator import evaluate_network_modules


def load_dict_from_py_file(filepath: str, dict_name: str = "SEED_MAP"):
    """Dynamically loads a python file and extracts a specific dictionary."""
    spec = importlib.util.spec_from_file_location("module.name", filepath)
    loaded_module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = loaded_module
    spec.loader.exec_module(loaded_module)

    # Extract the dictionary
    return getattr(loaded_module, dict_name)


def parse_filename(filename):
    """
    Safely extracts the original network name and metadata.
    Defaults to baseline if no noise tags are found.
    """
    # Strip the file extension
    if filename.endswith(".txt"):
        base_string = filename[:-4]
    elif filename.endswith(".tsv"):
        base_string = filename[:-4]
    else:
        base_string = filename

    # modification strings
    pert_types = [
        "targeted_hub_addition",
        "targeted_hub_removal",
        "targeted_periphery_addition",
        "targeted_periphery_removal",
        "added_edges",
        "removed_edges",
    ]

    pert_type = "baseline"
    network_name = base_string  # Default: the whole name is the network name

    # Find the perturbation type and split the string to get the network name
    for p_type in pert_types:
        # We look for _{p_type} to ensure we split at the exact right underscore
        if f"_{p_type}" in base_string:
            pert_type = p_type
            network_name = base_string.split(f"_{p_type}")[0]
            break

    # Extract noise and repeat
    noise_match = re.search(r"_noise(\d+(?:p\d+)?)", base_string)
    repeat_match = re.search(r"_repeat(\d+)", base_string)

    noise_level = float(noise_match.group(1).replace("p", ".")) if noise_match else 0.0
    repeat_num = int(repeat_match.group(1)) if repeat_match else 0

    return network_name, pert_type, noise_level, repeat_num


def precompute_shortest_paths(G):
    """Your highly optimized APSP function."""
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj_matrix = nx.to_scipy_sparse_array(G, nodelist=nodes)

    # Pro-tip: return as float32 to cut RAM usage from 2.6GB to 1.3GB!
    dist_matrix = shortest_path(
        csgraph=adj_matrix, directed=False, unweighted=True
    ).astype(np.float32)
    return dist_matrix, node_to_idx


def run_batch_pipeline(
    network_folders: list[str], modules_filepath: str, output_csv: str
):
    """
    Takes a list of directories, finds all networks, evaluates them, and exports a master CSV.
    """
    modules = load_dict_from_py_file(modules_filepath)
    all_results = []
    network_files = []

    # 1. Gather all files from the provided folders
    for folder in network_folders:
        folder_path = Path(folder)
        # Grab both .txt and .tsv depending on how you saved them
        network_files.extend(list(folder_path.rglob("*.txt")))
        network_files.extend(list(folder_path.rglob("*.tsv")))

    # Remove duplicates just in case paths overlap
    network_files = list(set(network_files))
    print(f"Discovered {len(network_files)} total networks to process.")

    # 2. Process each network sequentially
    for filepath in tqdm(network_files, desc="Processing Networks"):
        # Unpack all 4 variables directly from the new parse function
        network_name, pert_type, noise, rep = parse_filename(filepath.name)

        metadata = {
            "network_id": network_name,
            "perturbation_type": pert_type,
            "noise_level": noise,
            "repeat": rep,
        }

        # Read the graph
        G_pert = nx.read_edgelist(filepath)

        # Calculate APSP ONCE for this network
        dist_matrix, node_to_idx = precompute_shortest_paths(G_pert)

        # Evaluate all diseases + all nulls
        results = evaluate_network_modules(
            G_pert, modules, metadata, dist_matrix, node_to_idx
        )
        all_results.extend(results)

        # Explicit garbage collection to prevent 2.6GB memory stacking
        if results:
            df = pd.DataFrame(results)
            cols = [
                "network_id",
                "perturbation_type",
                "noise_level",
                "repeat",
                "disease",
            ]
            remaining_cols = [c for c in df.columns if c not in cols]
            df = df[cols + remaining_cols]

            # If the file doesn't exist, write it with headers. If it does, append without headers.
            df.to_csv(
                output_csv, mode="a", header=not os.path.exists(output_csv), index=False
            )

        del G_pert, dist_matrix, node_to_idx
