"""
Do betweenness-targeted and hub-targeted removal attack the same edges?

For the five core networks and the extended set, compares per edge:
  edge betweenness                (weight of targeted_betweenness_removal)
  product of the endpoint degrees (weight of targeted_hub_removal; the
                                   periphery weight is its inverse)
Reported per network:
  spearman           rank correlation of the two weights
  top10_overlap      share of the top 10% edges (the ones removed first) that
                     both rankings have in common; random rankings give ~0.10
  top1pct_mass_betw  share of the total weight on the top 1% of edges,
  top1pct_mass_hub   i.e. how concentrated the removal probability is

Networks are loaded as simple graphs (self-loops and multi-edges dropped).

Writes outputs/perturbation_weights/betweenness_vs_hub.csv
Run from the repository root:
  uv run python scripts/perturbed_networks/compare_betweenness_hub_weights.py
"""

import time

import igraph as ig
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASELINE_DIR = "data/baseline_networks/"
OUT = "outputs/perturbation_weights/betweenness_vs_hub.csv"

CORE = {
    "ppi (lab)": "ppi.csv",
    "huri": "huri.csv",
    "power": "power.csv",
    "astro": "astro.csv",
    "wiki": "wiki.csv",
}


def compare(path):
    df = pd.read_csv(path, header=None, names=["s", "t"], dtype=str)
    g = ig.Graph.TupleList(df.itertuples(index=False), directed=False).simplify()
    betweenness = np.array(g.edge_betweenness(directed=False))
    degree = np.array(g.degree())
    edges = np.array(g.get_edgelist())
    hub = degree[edges[:, 0]] * degree[edges[:, 1]]

    k = max(1, int(0.1 * len(betweenness)))
    top_b = set(np.argsort(-betweenness, kind="stable")[:k])
    top_h = set(np.argsort(-hub, kind="stable")[:k])
    k1 = max(1, len(betweenness) // 100)
    return {
        "N": g.vcount(),
        "E": g.ecount(),
        "spearman": spearmanr(betweenness, hub).statistic,
        "top10_overlap": len(top_b & top_h) / k,
        "top1pct_mass_betw": np.sort(betweenness)[::-1][:k1].sum() / betweenness.sum(),
        "top1pct_mass_hub": np.sort(hub)[::-1][:k1].sum() / hub.sum(),
    }


if __name__ == "__main__":
    overview = pd.read_csv(BASELINE_DIR + "extended_set/extended_set_overview.csv")
    overview = overview[overview.status == "ok"]
    networks = [(name, BASELINE_DIR + f, "core") for name, f in CORE.items()] + [
        (r.key, BASELINE_DIR + f"extended_set/{r.key}.csv", r.domain)
        for r in overview.itertuples()
    ]
    rows = []
    for name, path, domain in networks:
        t = time.time()
        rows.append({"network": name, "domain": domain, **compare(path)})
        print(f"{name:24s} rho={rows[-1]['spearman']:6.3f} ({time.time() - t:.0f}s)", flush=True)
    pd.DataFrame(rows).round(3).to_csv(OUT, index=False)
