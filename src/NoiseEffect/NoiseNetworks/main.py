import networkx as nx
import random
import numpy as np
import pandas as pd
import gzip
from pathlib import Path

##################################################
# Code generates networks with artificially,
# randomly introduced and removed edges
##################################################


# 1. Main function
def generateNoiseNetworksFromBaseline(
    path_to_edgelist: str,
    folder_to_save_perturbed: str,
    noise_levels: list[float],
    path_to_nodelist: str | None = None,
    noise_types: list[str] = [
        "added_edges",
        "removed_edges",
        "targeted_hub_addition",
        "targeted_hub_removal",
        "targeted_periphery_addition",
        "targeted_periphery_removal",
    ],
    num_repeats_per_noise_level: int = 10,
    network_name: str = "network",
):
    """
    Orchestrates the process of generating networks with randomly
    added and removed edges from a baseline network.

    :param path_to_edgelist: Path to the edgelist file of the baseline network
    :type path_to_edgelist: str
    :param folder_to_save_perturbed: Folder where perturbed networks will be saved
    :type folder_to_save_perturbed: str
    :param noise_levels: List of noise levels to apply
    :type noise_levels: list[float]
    :param path_to_nodelist: Optional path to a nodelist file. If provided, it will be used to add degree-0 nodes not present in an edgelist.
    :type path_to_nodelist: str | None
    :param noise_types: List of noise types to apply. Optional, defaults to all types
    :type noise_types: list[str]
    :param num_repeats_per_noise_level: Number of repeats per noise level. Optional, defaults to 10
    :type num_repeats_per_noise_level: int
    :param network_name: Information on the main network, will be included in the file name of the perturbed networks. Optional, defaults to "network"
    :type network_name: str
    """
    # Load the baseline network
    g, idx_to_node, node_to_idx = _loadBaseline(path_to_edgelist, path_to_nodelist)
    graph_info = {
        "degrees": dict(g.degree()),
        "nodes": list(g.nodes()),
        "edges": list(g.edges())
    }
    for noise_level in noise_levels:
        num_edges_to_modify = _calcualteNumberOfEdgesToModify(
            g, noise_level, noise_types, graph_info
        )
        for noise_type in noise_types:
            # List to hold the repeat results
            all_dfs = []
            for repeat in range(num_repeats_per_noise_level):
                # Generate perturbed graph based on the action word
                if "add" in noise_type:
                    G_perturbed = _addEdgesToNetwork(g, num_edges_to_modify, noise_type, graph_info)
                elif "remov" in noise_type:
                    G_perturbed = _removeEdgesFromNetwork(
                        g, num_edges_to_modify, noise_type, graph_info
                    )
                else:
                    raise ValueError(f"Unknown noise type: {noise_type}")

                # Extract edges and convert to DataFrame
                df_repeat = pd.DataFrame(G_perturbed.edges(), columns=["source", "target"])

                # Relabel nodes in the DataFrame back to original labels
                df_repeat["source"] = df_repeat["source"].map(idx_to_node)
                df_repeat["target"] = df_repeat["target"].map(idx_to_node)

                # Add the repeat ID as a column
                df_repeat["repeat"] = repeat

                # Store this DataFrame
                all_dfs.append(df_repeat)

            # Combine all repeats into single DataFrame
            final_df = pd.concat(all_dfs, ignore_index=True)

            _saveParquet(
                final_df,
                folder_to_save_perturbed,
                noise_level,
                network_name,
                modification_type=noise_type,
            )


############################################################
# 2. Loading the baseline network


def _loadBaseline(path_to_edgelist: str, path_to_nodelist: str | None = None):
    """
    Loads the baseline network from an edgelist file and creates
    mappings between original node labels and integer indices.
    Returns a networkx graph with integer-labeled nodes.

    :param path_to_edgelist: Path to the edgelist file of the baseline network
    :type path_to_edgelist: str
    """
    # Required params: path (file path to edgelist)
    g_raw = nx.read_edgelist(path_to_edgelist)
    # optionally add degree-0 nodes
    if path_to_nodelist is not None:
        with open(path_to_nodelist) as f:
            content = f.read().strip()
            all_nodes = content.split(",") if content else []
        g_raw.add_nodes_from(n.strip() for n in all_nodes if n.strip())
    # Ensure all nodes are strings
    if not all(isinstance(n, str) for n in g_raw.nodes()):
       raise TypeError("All node labels must be strings; got mixed/non-string types.")
    # Show the full node set was loaded
    print(f"Loaded {g_raw.number_of_nodes()} nodes and {g_raw.number_of_edges()} edges from the edgelist.")
    # Sorting the nodes to make the mapping always the same
    original_node_labels = sorted(list(g_raw.nodes()))
    # Create the forward and reverse mappings
    idx_to_node = {i: label for i, label in enumerate(original_node_labels)}
    node_to_idx = {label: i for i, label in enumerate(original_node_labels)}
    # Creating an integer-labeled graph
    g = nx.relabel_nodes(g_raw, node_to_idx)
    return g, idx_to_node, node_to_idx


############################################################
# 3. Introduce noise by adding and removing edges


def _addEdgesToNetwork(g: nx.Graph, num_edges_to_modify: int, noise_type: str, graph_info: dict):
    """
    Overall function to introduce noise by adding edges.

    :param g: Original network
    :type g: nx.Graph
    :param num_edges_to_modify: Number of edges to add and remove
    :type num_edges_to_modify: int
    """
    # Map the string to the correct function and target
    dispatch = {
        "added_edges": lambda: _randomEdgesToAdd(g, num_edges_to_modify, graph_info),
        "targeted_hub_addition": lambda: _targetedAddition(
            g, num_edges_to_modify, target="hubs", graph_info=graph_info
        ),
        "targeted_periphery_addition": lambda: _targetedAddition(
            g, num_edges_to_modify, target="periphery", graph_info=graph_info
        ),
    }

    if noise_type not in dispatch:
        raise ValueError(f"Addition strategy for '{noise_type}' is not defined.")

    # Execute the mapped function
    edges_to_add = dispatch[noise_type]()

    # Add the new edges to the graph
    G_added = g.copy()
    G_added.add_edges_from(edges_to_add)
    return G_added


def _removeEdgesFromNetwork(g: nx.Graph, num_edges_to_modify: int, noise_type: str, graph_info: dict):
    """
    Overall function to introduce noise by removing edges.

    :param g: Original network
    :type g: nx.Graph
    :param num_edges_to_modify: Number of edges to add and remove
    :type num_edges_to_modify: int
    """
    dispatch = {
        "removed_edges": lambda: _randomEdgesToRemove(g, num_edges_to_modify, graph_info),
        "targeted_hub_removal": lambda: _targetedRemoval(
            g, num_edges_to_modify, target="hubs", graph_info=graph_info
        ),
        "targeted_periphery_removal": lambda: _targetedRemoval(
            g, num_edges_to_modify, target="periphery", graph_info=graph_info
        ),
    }

    if noise_type not in dispatch:
        raise ValueError(f"Removal strategy for '{noise_type}' is not defined.")

    edges_to_remove = dispatch[noise_type]()

    G_removed = g.copy()
    G_removed.remove_edges_from(edges_to_remove)
    return G_removed


def _calcualteNumberOfEdgesToModify(g, noise_level, noise_types, graph_info):
    """
    Calculates how many edges shall be modified based on the noise level.

    :param g: Original network
    :type g: nx.Graph
    :param noise_level: Noise level to apply
    :type noise_level: float
    :param noise_types: Types of noise to apply (e.g., ["added_edges", "removed_edges"])
    :type noise_types: list[str]
    :param graph_info: Precomputed graph information (e.g., degrees, nodes, edges)
    :type graph_info: dict
    """

    num_edges = len(graph_info["edges"])
    num_modify = int(num_edges * noise_level)  # Number of edges to modify

    # Check if results make sense
    if num_modify <= 0:
        raise ValueError("Number of edges to modify can not be negative or zero.")
    if "removed_edges" in noise_types and num_modify >= num_edges:
        raise ValueError("Can not remove all edges in the graph.")

    return num_modify


############################################################
# 5. Random noise introduction


def _randomEdgesToAdd(g: nx.Graph, num_modify: int, graph_info: dict):
    """
    Random edge addition using rejection sampling.
    """
    nodes = graph_info["nodes"]
    num_nodes = len(nodes)
    edges_to_add = set()

    # Calculate max possible non-edges
    max_possible_edges = (num_nodes * (num_nodes - 1)) // 2 - g.number_of_edges()
    if num_modify > max_possible_edges:
        raise ValueError("More edges to be added than available non-edges.")

    # Loop until we have enough unique edges
    while len(edges_to_add) < num_modify:
        # Sample in batches to make numpy lightning fast
        batch_size = (num_modify - len(edges_to_add)) * 2

        # Pick random nodes uniformly
        u_indices = np.random.randint(0, num_nodes, size=batch_size)
        v_indices = np.random.randint(0, num_nodes, size=batch_size)

        for u_idx, v_idx in zip(u_indices, v_indices):
            if u_idx != v_idx:  # No self-loops
                u, v = nodes[u_idx], nodes[v_idx]

                if not g.has_edge(u, v):
                    # Sort to ensure (u,v) and (v,u) are treated identically by the set
                    edge = (u, v) if u < v else (v, u)
                    edges_to_add.add(edge)

                    if len(edges_to_add) == num_modify:
                        break

    return list(edges_to_add)


def _randomEdgesToRemove(g, num_modify, graph_info):
    """
    Gets all edges in the original graph and randomly samples edges to remove from the graph.

    :param g: Original network
    :type g: nx.Graph
    :param num_modify: Number of edges to remove
    :type num_modify: int
    :param graph_info: Precomputed graph information
    :type graph_info: dict
    """
    edges = graph_info["edges"]
    # Directly sample unique edges to remove
    edges_to_remove = random.sample(edges, num_modify)
    return edges_to_remove


############################################################
# 5. Targeted noise introduction


def _targetedRemoval(g: nx.Graph, num_modify: int, target: str, graph_info: dict):
    """
    Randomly samples edges to remove, weighting edges connected to
    high or low degree hubs depending on the target.
    """
    edges = graph_info["edges"]
    degrees = graph_info["degrees"]

    # Calculate weights based on the product of degrees
    if target == "hubs":
        weights = np.array([degrees[u] * degrees[v] for u, v in edges], dtype=float)
    elif target == "periphery":
        # max(..., 1) prevents DivisionByZero if isolated nodes exist.
        weights = np.array(
            [1.0 / (max(degrees[u], 1) * max(degrees[v], 1)) for u, v in edges],
            dtype=float,
        )

    # Normalize to create a probability distribution
    probabilities = weights / weights.sum()

    # Sample unique indices based on probabilities
    chosen_indices = np.random.choice(
        len(edges), size=num_modify, replace=False, p=probabilities
    )
    edges_to_remove = [edges[i] for i in chosen_indices]

    return edges_to_remove


def _targetedAddition(g: nx.Graph, num_modify: int, target: str, graph_info: dict):
    """
    Randomly samples non-edges to add, heavily weighting edges between
    high or low degree nodes depending on the target.
    """
    nodes = graph_info["nodes"]
    degrees = graph_info["degrees"]
    edges_to_add = set()

    max_possible_edges = (len(nodes) * (len(nodes) - 1)) // 2 - g.number_of_edges()
    if num_modify > max_possible_edges:
        raise ValueError("More edges to be added than available non-edges.")

    # 1. Calculate weights for INDIVIDUAL nodes, not edges!
    if target == "hubs":
        node_weights = np.array([degrees[n] for n in nodes], dtype=float)
    elif target == "periphery":
        node_weights = np.array([1.0 / max(degrees[n], 1) for n in nodes], dtype=float)
    else:
        raise ValueError("Target must be 'hubs' or 'periphery'")

    # Normalize to create probabilities
    node_probs = node_weights / node_weights.sum()

    # 2. Sample nodes based on those probabilities
    while len(edges_to_add) < num_modify:
        # Sample in batches (ask for 2x what we need to cover duplicates/rejections)
        batch_size = (num_modify - len(edges_to_add)) * 2

        u_batch = np.random.choice(nodes, size=batch_size, p=node_probs)
        v_batch = np.random.choice(nodes, size=batch_size, p=node_probs)

        for u, v in zip(u_batch, v_batch):
            if u != v and not g.has_edge(u, v):
                edge = (u, v) if u < v else (v, u)
                edges_to_add.add(edge)

                if len(edges_to_add) == num_modify:
                    break

    return list(edges_to_add)


############################################################
# 4. Save the perturbed edgelists


def _saveParquet(
    final_df: pd.DataFrame,
    folder_to_save_perturbed: str,
    noise_level: float,
    network_name: str,
    modification_type: str,
):
    """Saves the combined Pandas DataFrame as a Parquet file."""
    noise_level_str = str(noise_level).replace(".", "p")
    
    # Optional but recommended: convert to smaller data types to save RAM/Disk space
    # If repeat is always 0-99, uint8 is perfect
    final_df["repeat"] = final_df["repeat"].astype("uint8") 
    
    file_path = f"{folder_to_save_perturbed}/{network_name}_{modification_type}_noise_{noise_level_str}.parquet"
    final_df.to_parquet(file_path, index=False)
