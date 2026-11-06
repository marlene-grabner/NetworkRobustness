import gzip
import io
import os
import sys
import zipfile
from urllib.parse import quote

import networkx as nx
import pandas as pd
import requests

from catalog import NETWORKS

##################################################
# Downloads the networks listed in catalog.py from
# Netzschleuder / SNAP and brings them into the format
# used for the baseline networks:
#   - undirected, unweighted, simple (no self-loops, no multi-edges)
#   - giant connected component only
#   - comma separated, no header, string node labels
#
# Run from the repository root:
#   python scripts/baselines/fetch_extended_set.py            # all networks
#   python scripts/baselines/fetch_extended_set.py cora huri  # only these keys
##################################################

OUTPUT_FOLDER = "./data/baseline_networks/extended_set"
RAW_FOLDER = f"{OUTPUT_FOLDER}/original_download"
OVERVIEW_FILE = f"{OUTPUT_FOLDER}/extended_set_overview.csv"

# Size window applied AFTER simplification and GCC extraction
MIN_NODES = 1_000
MAX_EDGES = 150_000


############################################################
# 1. Downloading


def _downloadUrl(entry: dict) -> str:
    """Returns the download url of a catalog entry."""
    if entry["source"] == "netzschleuder":
        return f"https://networks.skewed.de/net/{entry['dataset']}/files/{quote(entry['net'])}.csv.zip"
    elif entry["source"] == "snap":
        return f"https://snap.stanford.edu/data/{entry['file']}"
    raise ValueError(f"Unknown source: {entry['source']}")


def _download(entry: dict, raw_folder: str = RAW_FOLDER) -> str:
    """
    Downloads the raw file of a catalog entry into raw_folder.
    Files which were already downloaded are not fetched again.
    """
    url = _downloadUrl(entry)
    suffix = ".csv.zip" if entry["source"] == "netzschleuder" else ".txt.gz"
    raw_path = f"{raw_folder}/{entry['key']}{suffix}"
    if os.path.exists(raw_path):
        return raw_path

    response = requests.get(url, timeout=300)
    response.raise_for_status()
    with open(raw_path, "wb") as f:
        f.write(response.content)
    return raw_path


############################################################
# 2. Reading the raw edgelists


def _readRawEdges(entry: dict, raw_path: str) -> pd.DataFrame:
    """
    Reads the first two columns (source, target) of a raw download as strings.
    Edge attributes (weights, timestamps, ...) are dropped.
    """
    if entry["source"] == "netzschleuder":
        # edges.csv starts with a header line like "# source, target, weight"
        with zipfile.ZipFile(raw_path) as zf:
            with zf.open("edges.csv") as f:
                df = pd.read_csv(
                    io.TextIOWrapper(f, encoding="utf-8"),
                    skiprows=1,
                    header=None,
                    usecols=[0, 1],
                    names=["source", "target"],
                    dtype=str,
                    skipinitialspace=True,
                )
    else:
        # SNAP: whitespace separated with '#' comment lines
        with gzip.open(raw_path, "rt") as f:
            df = pd.read_csv(
                f,
                sep=r"\s+",
                header=None,
                usecols=[0, 1],
                names=["source", "target"],
                comment="#",
                dtype=str,
            )
    df["source"] = df["source"].str.strip()
    df["target"] = df["target"].str.strip()
    return df


############################################################
# 3. Conversion into the baseline format


def _toBaselineGraph(df: pd.DataFrame) -> tuple[nx.Graph, dict]:
    """
    Symmetrizes, removes self-loops and multi-edges and extracts the GCC.
    Returns the GCC and a dict with statistics of each step.
    """
    stats = {"raw_rows": len(df)}

    self_loops = df["source"] == df["target"]
    stats["self_loops_removed"] = int(self_loops.sum())
    df = df[~self_loops]

    # nx.Graph is undirected and simple, so direction and duplicates collapse here
    g = nx.Graph()
    g.add_edges_from(zip(df["source"], df["target"]))
    stats["nodes_simple"] = g.number_of_nodes()
    stats["edges_simple"] = g.number_of_edges()
    stats["num_components"] = nx.number_connected_components(g)

    gcc_nodes = max(nx.connected_components(g), key=len)
    gcc = g.subgraph(gcc_nodes).copy()
    stats["nodes_gcc"] = gcc.number_of_nodes()
    stats["edges_gcc"] = gcc.number_of_edges()
    stats["gcc_fraction"] = round(gcc.number_of_nodes() / g.number_of_nodes(), 4)
    return gcc, stats


def _saveAndVerify(gcc: nx.Graph, output_path: str):
    """
    Writes the GCC as a headerless csv and reads it back to check that
    nodes and edges are preserved exactly.
    """
    pd.DataFrame(list(gcc.edges()), columns=["source", "target"]).to_csv(
        output_path, index=False, header=False
    )

    with open(output_path, "rb") as f:
        g_new = nx.read_edgelist(f, delimiter=",")

    nodes_match = set(gcc.nodes()) == set(g_new.nodes())
    edges_match = set(frozenset(e) for e in gcc.edges()) == set(
        frozenset(e) for e in g_new.edges()
    )
    if not (nodes_match and edges_match):
        os.remove(output_path)
        raise RuntimeError(f"Network structure changed when writing {output_path}")


############################################################
# 4. Run the pipeline


def processNetwork(
    entry: dict, output_folder: str = OUTPUT_FOLDER, raw_folder: str = RAW_FOLDER
) -> dict:
    """Downloads, converts and saves one catalog entry. Returns its overview row."""
    row = {k: entry.get(k) for k in ["key", "domain", "subdomain", "source", "dataset", "net", "file", "directed"]}
    row["url"] = _downloadUrl(entry)

    raw_path = _download(entry, raw_folder)
    df = _readRawEdges(entry, raw_path)
    gcc, stats = _toBaselineGraph(df)
    row.update(stats)

    if stats["nodes_gcc"] < MIN_NODES:
        row["status"] = f"skipped: GCC has fewer than {MIN_NODES} nodes"
    elif stats["edges_gcc"] > MAX_EDGES:
        row["status"] = f"skipped: GCC has more than {MAX_EDGES} edges"
    else:
        _saveAndVerify(gcc, f"{output_folder}/{entry['key']}.csv")
        row["status"] = "ok"
    return row


if __name__ == "__main__":
    os.makedirs(RAW_FOLDER, exist_ok=True)

    selected_keys = sys.argv[1:]
    entries = [e for e in NETWORKS if not selected_keys or e["key"] in selected_keys]
    unknown = set(selected_keys) - {e["key"] for e in NETWORKS}
    if unknown:
        raise ValueError(f"Unknown network keys: {unknown}")

    rows = []
    for entry in entries:
        print(f"Processing {entry['key']}...")
        try:
            row = processNetwork(entry)
        except Exception as e:
            row = {"key": entry["key"], "domain": entry["domain"], "status": f"failed: {e}"}
        print(f"  {row['status']} (N={row.get('nodes_gcc')}, E={row.get('edges_gcc')})")
        rows.append(row)

    # Update the overview, replacing rows of networks processed in this run
    # and dropping networks which are no longer in the catalog
    overview = pd.DataFrame(rows)
    if os.path.exists(OVERVIEW_FILE):
        previous = pd.read_csv(OVERVIEW_FILE, dtype={"net": str})
        catalog_keys = {e["key"] for e in NETWORKS}
        previous = previous[
            ~previous["key"].isin(overview["key"]) & previous["key"].isin(catalog_keys)
        ]
        overview = pd.concat([previous, overview], ignore_index=True)
    order = {e["key"]: i for i, e in enumerate(NETWORKS)}
    overview = overview.sort_values("key", key=lambda s: s.map(order))
    overview.to_csv(OVERVIEW_FILE, index=False)

    ok = overview[overview["status"] == "ok"]
    print(f"\n{len(ok)} networks in the extended set:")
    print(ok.groupby("domain").size().to_string())
