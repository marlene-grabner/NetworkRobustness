import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List

from .io_utils import get_universal_node_mapping, load_adjacency, parse_filename
from .spectral_math import (
    to_normalized_laplacian,
    get_spectral_components,
    get_eigenvector_centrality,
)
from .topology_math import (
    extract_gcc,
    calculate_sampled_shortest_path,
    calculate_modularity,
)
from .comparators import calculate_string_based_overlap, calculate_davis_kahan_bounds


def analyze_experiment(
    original_network: str,
    perturbed_directory: str,
    output_csv: str,
    network_label: str = "Unknown",
) -> pd.DataFrame:
    """
    The main orchestration engine. Processes a baseline network against a directory
    of perturbed repeats, calculating all spectral and topological metrics.
    """
    results = []

    # Pre-calculate baseline original network metrics (isolated)
    dummy_mapping = get_universal_node_mapping(original_network, original_network)
    A_orig_base = load_adjacency(original_network, dummy_mapping)

    print(
        f"[{network_label}] Calculating baseline topology (Modularity & Shortest Paths)..."
    )
    orig_modularity = calculate_modularity(A_orig_base)
    orig_shortest_path = calculate_sampled_shortest_path(A_orig_base, sample_size=500)

    search_pattern = os.path.join(
        perturbed_directory, "*noise*.txt"
    )  # Adjust extension if needed
    pert_files = glob.glob(search_pattern)
    print(
        f"[{network_label}] Found {len(pert_files)} perturbed networks. Processing..."
    )

    # Wrap pert_files in tqdm to generate the progress bar!
    # We use desc to give the bar a nice title.
    for pert_file in tqdm(pert_files, desc=f"Processing {network_label}"):
        # 1. Extract Metadata
        meta = parse_filename(pert_file)
        noise_type = meta["Noise_Type"]
        noise_level = meta["Noise_Level"]

        try:
            # ---> REMOVE OR COMMENT OUT THIS LINE so it doesn't break the bar visually <---
            # print(f"Processing: {noise_type} noise at {noise_level}...")

            # 2. Data Ingestion
            node_to_idx_univ = get_universal_node_mapping(original_network, pert_file)
            A_orig = load_adjacency(original_network, node_to_idx_univ)
            A_pert = load_adjacency(pert_file, node_to_idx_univ)

            # 3. Eigenvector Centrality (Global Hubs)
            cent_orig = get_eigenvector_centrality(A_orig)
            cent_pert = get_eigenvector_centrality(A_pert)
            overlap_cent = calculate_string_based_overlap(
                cent_orig, cent_pert, node_to_idx_univ, node_to_idx_univ, k=200
            )

            # 4. Topology Math (GCC, Path Length, Modularity)
            A_gcc_orig, dict_gcc_orig, _ = extract_gcc(A_orig, node_to_idx_univ)
            A_gcc_pert, dict_gcc_pert, gcc_fraction_pert = extract_gcc(
                A_pert, node_to_idx_univ
            )

            pert_modularity = calculate_modularity(A_pert)
            pert_shortest_path = calculate_sampled_shortest_path(
                A_pert, sample_size=500
            )

            if A_gcc_pert.shape[0] < 5:
                print(f"  -> Skipping {meta['Filename']}: GCC shattered.")
                continue

            # 5. Spectral Math (Fiedler Vector & Gap)
            L_gcc_orig = to_normalized_laplacian(A_gcc_orig)
            L_gcc_pert = to_normalized_laplacian(A_gcc_pert)

            vals_orig, vecs_orig = get_spectral_components(L_gcc_orig, k=5)
            vals_pert, vecs_pert = get_spectral_components(L_gcc_pert, k=5)

            overlap_fiedler = calculate_string_based_overlap(
                vecs_orig[:, 1], vecs_pert[:, 1], dict_gcc_orig, dict_gcc_pert, k=200
            )

            pert_spectral_gap = vals_pert[2] - vals_pert[1]
            orig_spectral_gap = vals_orig[2] - vals_orig[1]

            # 6. Davis-Kahan Bounds (Only strictly valid for edge additions)
            dk_metrics = {
                "Norm_Delta_L": np.nan,
                "DK_Theoretical_Bound": np.nan,
                "Empirical_Shift_SinTheta": np.nan,
            }
            if noise_type == "added":
                L_full_orig = to_normalized_laplacian(A_orig)
                L_full_pert = to_normalized_laplacian(A_pert)
                dk_metrics = calculate_davis_kahan_bounds(
                    L_full_orig, L_full_pert, vals_orig, vecs_orig, vecs_pert
                )

            # 7. Compile Data Row
            row = {
                "Network_Model": network_label,
                "Noise_Type": noise_type,
                "Noise_Level": noise_level,
                "Repeat_ID": meta["Repeat_ID"],
                "Overlap_Centrality": overlap_cent,
                "Overlap_Fiedler": overlap_fiedler,
                "GCC_Fraction": gcc_fraction_pert,
                "Modularity_Q": pert_modularity,
                "Shortest_Path_Avg": pert_shortest_path,
                "Spectral_Gap": pert_spectral_gap,
                "Orig_Modularity_Q": orig_modularity,
                "Orig_Shortest_Path": orig_shortest_path,
                "Orig_Spectral_Gap": orig_spectral_gap,
            }
            # Unpack Davis-Kahan metrics into the row
            row.update(dk_metrics)
            results.append(row)

        except Exception as e:
            print(f"  -> Error processing {meta['Filename']}: {e}")

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results.sort_values(
            by=["Noise_Type", "Noise_Level", "Repeat_ID"], inplace=True
        )
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        print(f"[{network_label}] Saved {len(df_results)} rows to {output_csv}")

    return df_results
