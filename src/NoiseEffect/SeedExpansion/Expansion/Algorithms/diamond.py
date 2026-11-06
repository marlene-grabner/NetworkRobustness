"""
DIAMOnD (DIseAse MOdule Detection), Ghiassian et al. 2015, reimplemented on a
CSR adjacency matrix instead of networkx.

At each iteration, every non-module node is scored by the hypergeometric
p-value of its connectivity to the current module (given its degree and the
module's size), and the most significant node is added. Only nodes with a
non-zero connection to the module are ever candidates, so we track a
`candidates` dict of {node_idx: links_to_module} and update it incrementally
(only touching neighbors of the node just added) rather than rescanning the
whole graph every iteration -- this is the key optimization from the original
implementation and is what makes it tractable at scale.

DIAMOnD does not naturally produce a *full* ranking (it's a greedy sequential
process), so the score vector returned is:
    max_added_nodes - order_added   for the first `max_added_nodes` nodes chosen
    -inf                            for every other node (seeds included)
This is sufficient for top-k comparisons as long as k <= max_added_nodes.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import hypergeom


def diamond_score(adj: sparse.csr_matrix, seed_idx: np.ndarray, max_added_nodes: int = 200, alpha: float = 1.0) -> np.ndarray:
    """
    DIAMOnD algorithm optimized with vectorized hypergeom pair evaluation 
    and fast CSR neighborhood pointer tracking.
    
    Deterministic tie-breaking is enforced via global matrix coordinate sorting.
    """
    n_nodes = adj.shape[0]

    # Track which nodes are inside our growing module
    in_module = np.zeros(n_nodes, dtype=bool)
    in_module[seed_idx] = True
    
    # Pre-calculate total degrees for all nodes in the network (k_total)
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    
    # Track connections to original seeds permanently for alpha scaling
    conn_to_seeds = np.asarray(adj[seed_idx, :].sum(axis=0)).flatten()
    
    # Track dynamic total connections from each node to the growing module
    conn_to_module = conn_to_seeds.copy()
    
    # Initialize unadded nodes to 0 (finite, makes them eligible negatives)
    scores = np.zeros(n_nodes)
    # Set the seeds to -inf (forces metrics to ignore them)
    scores[seed_idx] = -np.inf

    # Initialize a live set of candidates to avoid O(N) rescans
    candidate_set = set(np.where((~in_module) & (conn_to_module > 0))[0])
    
    # Pre-calculate baseline alpha constants
    s0_base = len(seed_idx)
    s0_inflated = s0_base * alpha
    N_adjusted = n_nodes + (s0_inflated - s0_base)
    
    for step in range(max_added_nodes):
        if not candidate_set:
            break
        
        # To enforce determinism, convert set to array and sort it
        # so candidate order maps to static global IDs
        candidates = np.array(list(candidate_set), dtype=np.int64)
        candidates.sort() 

        # Pull parameters for all active candidates
        k_in = conn_to_module[candidates]
        k_total = degrees[candidates]
        seed_edges = conn_to_seeds[candidates]
        
        # alpha scaling equations
        # Current module size = inflated seeds + normally added iteration nodes
        s0_adjusted = s0_inflated + step
        k_in_adjusted = k_in + (alpha - 1) * seed_edges
        k_total_adjusted = k_total + (alpha - 1) * seed_edges

        # Sort candidates by k_total_adjusted (primary) and k_in_adjusted (secondary)
        sort_order = np.lexsort((k_in_adjusted, k_total_adjusted))
        sorted_kt = k_total_adjusted[sort_order]

        # The last element of each contiguous k_total block has the max k_in
        mask = np.empty(len(sorted_kt), dtype=bool)
        mask[:-1] = sorted_kt[:-1] != sorted_kt[1:]
        mask[-1] = True

        max_k_in_indices = sort_order[mask]

        # Compute hypergeom.sf only on max k_in candidates per degree
        p_reduced = hypergeom.sf(
            k_in_adjusted[max_k_in_indices] - 1,
            N_adjusted,
            s0_adjusted,
            k_total_adjusted[max_k_in_indices],
        )

        # Identify winning degree pair(s) with the minimum p-value
        min_p = np.min(p_reduced)
        winning_mask = p_reduced == min_p
        win_k_in = k_in_adjusted[max_k_in_indices[winning_mask]]
        win_k_tot = k_total_adjusted[max_k_in_indices[winning_mask]]

        # Match winning pair(s) back to candidate array
        is_winner = np.zeros(len(candidates), dtype=bool)
        for w_in, w_tot in zip(win_k_in, win_k_tot):
            is_winner |= (k_in_adjusted == w_in) & (k_total_adjusted == w_tot)

        # Find the best candidate
        # np.argmax selects the FIRST True element (lowest global candidate index)
        best_idx = candidates[np.argmax(is_winner)]

        # Add them to the module
        in_module[best_idx] = True
        scores[best_idx] = max_added_nodes - step  # Higher score = added earlier

        # Remove the winner from the live candidate pool
        candidate_set.remove(best_idx)
        
        # Optimized low-level CSR neighborhood lookup
        start_ptr = adj.indptr[best_idx]
        end_ptr = adj.indptr[best_idx + 1]
        new_node_neighbors = adj.indices[start_ptr:end_ptr]

        # Vectorized connection update
        conn_to_module[new_node_neighbors] += 1
        
        # Incrementally add new valid neighbors to the live candidate pool
        for nb in new_node_neighbors:
            if not in_module[nb]:
                candidate_set.add(nb)
        
    return scores

"""
def diamond_score(adj: sparse.csr_matrix, seed_idx: np.ndarray, max_added_nodes: int = 200, alpha: float = 1.0) -> np.ndarray:
    n_nodes = adj.shape[0]
    
    in_module = np.zeros(n_nodes, dtype=bool)
    in_module[seed_idx] = True
    degrees = np.asarray(adj.sum(axis=1)).flatten()
    
    # 1. Track seed connections permanently (these never change)
    conn_to_seeds = np.asarray(adj[seed_idx, :].sum(axis=0)).flatten()
    
    # 2. Track total module connections (this grows)
    conn_to_module = conn_to_seeds.copy()
    
    scores = np.full(n_nodes, -np.inf)
    
    # Pre-calculate the TRUE inflated parameters once
    # We only inflate the seeds, and seeds never grow.
    s0_base = len(seed_idx)
    s0_inflated = s0_base * alpha
    N_inflated = n_nodes + (s0_inflated - s0_base)
    
    for step in range(max_added_nodes):
        candidates = np.where((~in_module) & (conn_to_module > 0))[0]
        if len(candidates) == 0:
            break
            
        k_in = conn_to_module[candidates]
        k_total = degrees[candidates]
        
        # 3. ONLY inflate the edges that actually connect to the seeds
        # conn_to_seeds[candidates] gives the specific seed-edges for these candidates
        seed_edges = conn_to_seeds[candidates]
        
        k_in_adjusted = k_in + (alpha - 1) * seed_edges
        k_total_adjusted = k_total + (alpha - 1) * seed_edges
        
        # The module size is the inflated seeds + the number of normally added nodes
        current_s0_adjusted = s0_inflated + step
        
        p_values = hypergeom.sf(k_in_adjusted - 1, N_inflated, current_s0_adjusted, k_total_adjusted)
        
        # 4. Break ties randomly to remove deterministic ID bias
        best_idx = candidates[np.argmin(p_values)]
        
        in_module[best_idx] = True
        scores[best_idx] = max_added_nodes - step 
        
        # 5. Update only the dynamic boundary connections
        new_node_neighbors = adj[best_idx, :].tocoo().col
        conn_to_module[new_node_neighbors] += 1
        
    return scores

def build_toy_graph():
    
    Builds a small 8-node CSR matrix:
    - Node 0 is the SEED.
    - Node 1: k_in=1, k_total=3  (Neighbors: 0, 4, 5)
    - Node 2: k_in=1, k_total=3  (Neighbors: 0, 4, 6)  <-- Ties with Node 1!
    - Node 3: k_in=1, k_total=4  (Neighbors: 0, 5, 6, 7)
    
    edges = [
        (0, 1), (0, 2), (0, 3), # Seed 0 connected to 1, 2, 3
        (1, 4), (1, 5),         # Node 1
        (2, 4), (2, 6),         # Node 2
        (3, 5), (3, 6), (3, 7)  # Node 3
    ]
    
    rows = [u for u, v in edges] + [v for u, v in edges]
    cols = [v for u, v in edges] + [u for u, v in edges]
    data = np.ones(len(rows), dtype=float)
    
    return sparse.csr_matrix((data, (rows, cols)), shape=(8, 8))


if __name__ == "__main__":
    adj = build_toy_graph()
    seed_idx = np.array([0])  # Start with Node 0 as seed
    
    final_scores = diamond_score(adj, seed_idx=seed_idx, max_added_nodes=3)
    
    print("\n================ FINAL RESULTS ================")
    print(f"Scores Array: {final_scores}")

"""