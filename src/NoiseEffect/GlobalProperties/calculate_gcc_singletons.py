import os
import glob
import pandas as pd
import igraph as ig
from concurrent.futures import ProcessPoolExecutor, as_completed


def true_baseline_node_count(baseline_path):
    df_base = pd.read_csv(
        baseline_path, sep="\t", header=None, names=["source", "target"]
    )
    nodes = set(df_base["source"]).union(set(df_base["target"]))
    # union in isolated nodes from the sidecar, if one exists
    sidecar = baseline_path.rsplit(".", 1)[0] + "_isolated_nodes.csv"
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            content = f.read().strip()
        if content:
            nodes |= {n.strip() for n in content.split(",") if n.strip()}
    return len(nodes)


def _process_singletons_and_gcc(file_path: str, total_baseline_nodes: int) -> list:
    """
    Worker function to process a single parquet file (100 repeats) for fast metrics.
    """
    print(f"Processing file: {file_path}")
    filename = os.path.basename(file_path)
    filename_base = filename.split(".")[0]
    df_pert = pd.read_parquet(file_path)

    if "repeat" not in df_pert.columns:
        raise ValueError(f"Parquet file {filename} must contain a 'repeat' column.")

    # Calculate num of singletons
    # ----------------------------
    # Melt source/target into one column, group by repeat, and count unique active nodes
    melted = df_pert[["repeat", "source", "target"]].melt(
        id_vars=["repeat"], value_name="node"
    )
    active_nodes_per_repeat = melted.groupby("repeat")["node"].nunique()

    results = []

    # Calculate GCC with igraph
    # ----------------------------
    for repeat_id, group in df_pert.groupby("repeat"):
        # Get pre-calculated singletons
        active_count = active_nodes_per_repeat.get(repeat_id, 0)
        num_singletons = total_baseline_nodes - active_count

        # Build igraph directly from the pandas dataframe
        # directed=False ensures an undirected graph
        # strips away the Pandas StringDtype which igraph can't work with
        edges = group[["source", "target"]].values.tolist()

        # Build igraph using TupleList, which natively parses string names
        g = ig.Graph.TupleList(edges, directed=False)

        # Calculate GCC fraction
        # g.components() computes all connected components; .sizes() gets their lengths
        if g.vcount() > 0:
            gcc_size = max(g.components().sizes())
        else:
            gcc_size = 0

        gcc_frac = gcc_size / total_baseline_nodes

        results.append(
            {
                "network_id": f"{filename_base}_repeat_{repeat_id}",
                "num_singletons": num_singletons,
                "gcc": gcc_frac,
            }
        )

    return results


def calculate_singletons_and_gcc(
    baseline_path: str, perturbed_dir: str, max_workers: int = None
) -> pd.DataFrame:
    """
    Calculate singletons and GCC using Multiprocessing.
    """
    # Load baseline efficiently to get node count
    total_baseline_nodes = true_baseline_node_count(baseline_path)

    parquet_files = glob.glob(os.path.join(perturbed_dir, "*.parquet"))
    all_results = []

    # Process files in parallel
    # If max_workers is None, it defaults to the number of processors on the machine.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to their respective file processing tasks
        futures = {
            executor.submit(
                _process_singletons_and_gcc, f_path, total_baseline_nodes
            ): f_path
            for f_path in parquet_files
        }

        for future in as_completed(futures):
            try:
                file_results = future.result()
                all_results.extend(file_results)
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")

    return pd.DataFrame(all_results)
