import graph_tool.all as gt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ====================================================================
# DISCLAIMER: This file only works with the graphtool package
# installed, which is not available in the current environment due
# to a mismatch of python version. Use graphtool-env instead
# ====================================================================

##################################################
# Load empirical networks
##################################################

networks_path = "./data/baseline_networks/"

G_ppi = nx.read_edgelist(networks_path + "chloe_ppi_lcc_2026_02_23.tsv", delimiter="\t")
G_power = nx.read_edgelist(networks_path + "western_us_power_grid.tsv", delimiter="\t")
G_collab = nx.read_edgelist(networks_path + "ca-AstroPh_gcc.tsv", delimiter="\t")
G_wiki = nx.read_edgelist(networks_path + "wiki-Vote_gcc.tsv", delimiter="\t")


# ====================================================================
# Functions
# ====================================================================


##################################################
# Make SBM null model
##################################################


def sbm_model_from_graph(G: nx.Graph, nested: bool = False) -> nx.Graph:
    """
    Fits a Stochastic Block Model (SBM) to the input graph G using graph-tool,
    and generates a randomized null model that preserves the inferred block structure.
    Parameters:
        - G: Input graph as a NetworkX graph object.
        - nested: If True, fits a hierarchical SBM; otherwise, fits a degree-corrected SBM.
    Returns:
        - null_model_nx: A NetworkX graph object representing the SBM null model.
        - state: The fitted graph-tool SBM state object (useful for diagnostics).
        - node_to_idx: A mapping from original NetworkX node IDs to graph-tool vertex indices (useful for block assignment verification).
    """
    #  Empty graphtool graph
    g = gt.Graph(directed=False)

    # Maooing between networkx noide IDs and graphtool ints
    original_nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(original_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    # Add original vertices
    g.add_vertex(len(original_nodes))

    # Add edges
    gt_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    g.add_edge_list(gt_edges)

    # Chose which type of SBM to fit
    if nested:
        # Hierachical SBM
        state = gt.minimize_nested_blockmodel_dl(g)
        lstate = state.get_levels()[0]
    else:
        # Degree corrected SBM
        state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=True))
        lstate = state

    # Statistics
    entropy_score = state.entropy()
    num_communities = lstate.get_nonempty_B()
    print(f"SBM Entropy score: {entropy_score:.4f}")
    print(f"SBM: Found {num_communities} active communities at the base layer.")

    # Extract parameters as native C-aligned arrays/matrices for generation
    b_array = lstate.b.a  # Extracts the block assignments as a raw NumPy array
    e_matrix = lstate.get_matrix()  # Returns the block edge count sparse matrix
    degrees = g.degree_property_map(
        "out"
    ).a  # Highly optimized C++ array fetch for degrees

    # Generate the randomized reference model graph
    # Both tracks now pass the degree sequence to ensure unified degree-correction constraints
    null_model_gt = gt.generate_sbm(
        b_array,
        e_matrix,
        out_degs=degrees,
        micro_ers=False,
        micro_degs=False,
        directed=False,
    )

    # Convert to networkx
    null_model_nx = nx.Graph()
    null_model_nx.add_nodes_from(original_nodes)

    # Translate graph-tool edge indicies to networkx node IDs
    null_edges = [
        (idx_to_node[int(e.source())], idx_to_node[int(e.target())])
        for e in null_model_gt.edges()
    ]
    null_model_nx.add_edges_from(null_edges)

    return null_model_nx, state, node_to_idx


def verify_sbm_blocks(G_real, G_sbm, state, node_to_idx, network_name, save_fig=None):
    """
    Verifies that the block-to-block mixing structure of the real network
    is accurately preserved in the generated SBM null model.
    Safely handles non-contiguous graph-tool block labels.
    """
    # 1. Extract raw block assignments from graph-tool state
    gt_blocks = state.get_blocks()

    node_to_block = {}
    for node, idx in node_to_idx.items():
        node_to_block[node] = int(gt_blocks[idx])

    # FIX: Map raw non-contiguous block labels to sequential matrix indices (0 to num_blocks-1)
    unique_blocks = sorted(list(set(node_to_block.values())))
    num_blocks = len(unique_blocks)
    block_to_matrix_idx = {block_id: i for i, block_id in enumerate(unique_blocks)}

    print(
        f"Verifying structure across {num_blocks} active blocks (Max Block ID found: {max(unique_blocks)})..."
    )

    # 2. Helper function to compute the block mixing matrix E_rs using mapped indices
    def compute_mixing_matrix(G, node_to_block, block_to_matrix_idx, num_blocks):
        matrix = np.zeros((num_blocks, num_blocks))
        for u, v in G.edges():
            b_u_raw = node_to_block.get(u)
            b_v_raw = node_to_block.get(v)

            if b_u_raw is not None and b_v_raw is not None:
                # Convert raw block IDs to sequential matrix indices
                b_u = block_to_matrix_idx[b_u_raw]
                b_v = block_to_matrix_idx[b_v_raw]

                matrix[b_u, b_v] += 1
                if b_u != b_v:
                    matrix[b_v, b_u] += 1  # Symmetric for undirected graphs
        return matrix

    # 3. Compute mixing matrices for both graphs
    E_real = compute_mixing_matrix(
        G_real, node_to_block, block_to_matrix_idx, num_blocks
    )
    E_sbm = compute_mixing_matrix(G_sbm, node_to_block, block_to_matrix_idx, num_blocks)

    # 4. Quantitative Check: Correlation of Edge Allocations
    tri_u_indices = np.triu_indices(num_blocks)
    real_flat = E_real[tri_u_indices]
    sbm_flat = E_sbm[tri_u_indices]

    r_val, _ = stats.pearsonr(real_flat, sbm_flat)
    print(f"--> Pearson correlation of block mixing matrices: r = {r_val:.4f}")

    if r_val > 0.95:
        print("SUCCESS: The mesoscale block-structure is highly preserved!")
    else:
        print("WARNING: Low correlation. The SBM did not capture the blocks correctly.")

    # 5. Visual Check: Plot Side-by-Side Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axes[0].imshow(np.log10(E_real + 1), cmap="viridis", origin="lower")
    axes[0].set_title(
        f"{network_name.capitalize().replace('_', ' ')}. Mixing Matrix (Log10)"
    )
    axes[0].set_xlabel("Sequential Block Index")
    axes[0].set_ylabel("Sequential Block Index")

    im2 = axes[1].imshow(np.log10(E_sbm + 1), cmap="viridis", origin="lower")
    axes[1].set_title("SBM Null Model Mixing Matrix (Log10)")
    axes[1].set_xlabel("Sequential Block Index")

    fig.colorbar(im2, ax=axes.ravel().tolist(), label="Log10(Edge Count + 1)")
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    plt.show()

    return r_val


# ====================================================================
# Results
# ====================================================================

##################################################
# Load empirical networks
##################################################

for name, G in zip(
    ["chloe_ppi", "western_us_power_grid", "ca-AstroPh", "wiki-Vote"],
    [G_ppi, G_power, G_collab, G_wiki],
):
    G_sbm, state_sbm, node_to_idx = sbm_model_from_graph(G)

    print(f"Original {name} network:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"\nSBM null model of {name} network:")
    print(f"Number of nodes: {G_sbm.number_of_nodes()}")
    print(f"Number of edges: {G_sbm.number_of_edges()}")

    nx.write_edgelist(
        G_sbm,
        f"./data/baseline_networks/null_models/{name}_sbm.tsv",
        delimiter="\t",
    )

    verify_sbm_blocks(
        G,
        G_sbm,
        state_sbm,
        node_to_idx,
        network_name=name,
        save_fig=f"outputs/figures/baseline_properties/sbm_confirmation/{name}_sbm_heatmap.pdf",
    )
