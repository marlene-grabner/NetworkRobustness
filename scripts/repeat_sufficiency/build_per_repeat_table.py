"""
Collect the per-repeat results of all scales into one long table.

Each row is one perturbation repeat of one curve: network, model, metric,
action, target, noise_level, repeat, value. A second table holds the
unperturbed value of each metric where one exists.

Writes:
  outputs/repeat_sufficiency/per_repeat_values.parquet
  outputs/repeat_sufficiency/baseline_values.csv

Run from the repository root:
  python scripts/repeat_sufficiency/build_per_repeat_table.py
"""

import glob
import os
import re

import pandas as pd

OUT_DIR = "outputs/repeat_sufficiency/"

NETWORKS = [
    f"{net}{model}"
    for net in ("ppi", "power", "astro", "wiki")
    for model in ("", "_er", "_conf", "_sbm")
]
TARGETS = ("hub", "periphery", "random")
COMMUNITY_ALGORITHMS = ("infomap", "leiden", "louvain", "label_propagation")
LOCAL_K = 50


def parse_action(text):
    if re.search(r"addition|added", text):
        return "addition"
    if re.search(r"removal|removed", text):
        return "removal"
    raise ValueError(f"no action in {text}")


def parse_noise(text):
    return float(re.search(r"noise_(\d+(?:p\d+)?)", text).group(1).replace("p", "."))


def split_network(network):
    for model in ("er", "conf", "sbm"):
        if network.endswith(f"_{model}"):
            return network[: -len(model) - 1], model
    return network, "empirical"


def load_gcc():
    rows, baselines = [], []
    for network in NETWORKS:
        path = f"outputs/global_properties/gcc_singletons/gcc_singletons_per_network/gcc_singletons_{network}.csv"
        df = pd.read_csv(path)
        baselines.append(("gcc", network, 1.0))
        df = df[df["perturbation_method"] != "baseline"]
        for network_id, gcc, method in zip(df["network_id"], df["gcc"], df["perturbation_method"]):
            rows.append(
                (network, "gcc", parse_action(network_id), method.split("_")[1],
                 parse_noise(network_id), int(network_id.rsplit("_", 1)[1]), gcc)
            )
    return rows, baselines


def load_fiedler():
    base = pd.read_csv(
        "outputs/global_properties/algebraic_connectivity/baseline/baseline_algebraic_connectivity.csv"
    ).set_index("network_id")["algebraic_connectivity"]
    rows, baselines = [], []
    for network in NETWORKS:
        baselines.append(("fiedler", network, float(base[network])))
        for target in TARGETS:
            folder = f"outputs/global_properties/algebraic_connectivity/perturbed/{network}/perturbed_{target}_target/"
            for path in glob.glob(folder + "*.csv"):
                df = pd.read_csv(path)
                name = os.path.basename(path)
                action, noise = parse_action(name), parse_noise(name)
                for network_id, value in zip(df["network_id"], df["algebraic_connectivity"]):
                    rows.append(
                        (network, "fiedler", action, target, noise,
                         int(network_id.rsplit("_", 1)[1]), value)
                    )
    return rows, baselines


def load_efficiency():
    df = pd.read_csv(
        "outputs/global_properties/global_efficiency/global_efficiency_baseline_and_perturbed.csv"
    )
    base = df[df["noise_type"] == "baseline"].set_index("network")["global_efficiency"]
    baselines = [("global_efficiency", n, float(v)) for n, v in base.items()]
    df = df[df["noise_type"] != "baseline"]
    rows = [
        (n, "global_efficiency", a, t, lvl, int(r), v)
        for n, t, a, lvl, r, v in zip(
            df["network"], df["noise_type"], df["action"], df["noise_level"],
            df["repeat"], df["global_efficiency"],
        )
    ]
    return rows, baselines


def load_community():
    rows, baselines = [], []
    for algorithm in COMMUNITY_ALGORITHMS:
        metric = f"ari_{algorithm}"
        for network in NETWORKS:
            baselines.append((metric, network, 1.0))
            for target in TARGETS:
                folder = f"outputs/local_structure/recovery_perturbed/{algorithm}/{network}/perturbed_{target}_target/"
                for path in glob.glob(folder + "*.csv"):
                    df = pd.read_csv(path)
                    name = os.path.basename(path)
                    action, noise = parse_action(name), parse_noise(name)
                    for repeat, value in zip(df["repeat_id"], df["vs_baseline_ari_mean"]):
                        rows.append((network, metric, action, target, noise, int(repeat), value))
    return rows, baselines


def load_local():
    """One value per repeat: the mean over the seeds at k = LOCAL_K."""
    frames = []
    columns = ["k", "is_shortest_k", "jaccard", "auprc", "network", "perturbation_type",
               "modification_type", "noise_level", "repeat", "algorithm"]
    for path in glob.glob("outputs/seed_expansion/expansion/perturbed_metrics/*.parquet"):
        df = pd.read_parquet(path, columns=columns)
        df = df[(df["k"] == LOCAL_K) & (~df["is_shortest_k"])]
        frames.append(
            df.groupby(
                ["network", "perturbation_type", "modification_type", "noise_level", "repeat", "algorithm"],
                as_index=False,
            )[["jaccard", "auprc"]].mean()
        )
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for measure in ("jaccard", "auprc"):
        rows += [
            (n, f"{measure}_{alg}", a, t.split("_")[0], lvl, int(r), v)
            for n, t, a, lvl, r, alg, v in zip(
                df["network"], df["perturbation_type"], df["modification_type"],
                df["noise_level"], df["repeat"], df["algorithm"], df[measure],
            )
        ]
    return rows, []


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, baselines = [], []
    for loader in (load_gcc, load_fiedler, load_efficiency, load_community, load_local):
        new_rows, new_baselines = loader()
        rows += new_rows
        baselines += new_baselines
        print(f"{loader.__name__}: {len(new_rows)} rows", flush=True)

    df = pd.DataFrame(
        rows, columns=["network", "metric", "action", "target", "noise_level", "repeat", "value"]
    )
    df[["base_network", "model"]] = [split_network(n) for n in df["network"]]
    df.to_parquet(OUT_DIR + "per_repeat_values.parquet", index=False)
    pd.DataFrame(baselines, columns=["metric", "network", "baseline"]).to_csv(
        OUT_DIR + "baseline_values.csv", index=False
    )

    n_repeats = df.groupby(["network", "metric", "action", "target", "noise_level"])["repeat"].nunique()
    print("repeats per noise level:", n_repeats.value_counts().to_dict())
