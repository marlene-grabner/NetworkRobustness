"""
Scans data/perturbed_networks/**/*.parquet and writes one job-table row per
parquet file (= one network x perturbation-target x noise-level combination;
each file internally holds all 100 repeats).

Expected layout, e.g.:
    data/perturbed_networks/astro/perturbed_hub_target/
        astro_targeted_hub_addition_noise_0p1.parquet

    -> network         = "astro"          (grandparent folder name)
    -> noise_type       = "hub"            ("perturbed_" and "_target" stripped)
    -> action           = "addition"       ("add" vs "remov" in the filename)
    -> noise_level      = 0.1              (parsed from "..._noise_0p1.parquet")

`network` (the folder name) must resolve to the same network name used for
the baseline job table, i.e. a file data/baseline_networks/<network>.csv or
data/baseline_networks/null_models/<network>.csv must exist.

Usage:
    uv run scripts/global_properties/global_efficiency/generate_perturbed_job_table.py
"""
import re
from pathlib import Path
import pandas as pd

PERTURBED_DIR = Path("data/perturbed_networks")
BASELINE_DIR = Path("data/baseline_networks")
NULL_MODEL_DIR = Path("data/baseline_networks/null_models")
OUT_PATH = Path("slurm_scripts/global_properties/global_efficiency/jobs/perturbed_jobs.csv")

NOISE_RE = re.compile(r"noise_([0-9p]+)\.parquet$")


def parse_noise_level(filename: str) -> float:
    m = NOISE_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse noise level from filename: {filename}")
    return float(m.group(1).replace("p", "."))


def resolve_baseline_path(network: str) -> str:
    base_path = BASELINE_DIR / f"{network}.csv"
    if base_path.exists():
        return str(base_path)
    null_model_path = NULL_MODEL_DIR / f"{network}.csv"
    if null_model_path.exists():
        return str(null_model_path)
    raise ValueError(
        f"No baseline edgelist found for network '{network}' "
        f"(looked in {BASELINE_DIR} and {NULL_MODEL_DIR})"
    )


def main():
    files = sorted(PERTURBED_DIR.glob("*/*/*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under {PERTURBED_DIR}/*/*/*.parquet")

    rows = []
    for f in files:
        network = f.parent.parent.name
        noise_type = f.parent.name.removeprefix("perturbed_").removesuffix("_target")

        if "remov" in f.name:
            action = "removal"
        elif "add" in f.name:
            action = "addition"
        else:
            raise ValueError(f"Could not classify addition vs removal for file: {f.name}")

        noise_level = parse_noise_level(f.name)
        edgelist_path = resolve_baseline_path(network)

        rows.append({
            "network": network,
            "noise_type": noise_type,
            "action": action,
            "noise_level": noise_level,
            "edgelist_path": edgelist_path,
            "parquet_path": str(f),
        })

    df = pd.DataFrame(rows).sort_values(["network", "noise_type", "action", "noise_level"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} jobs to {OUT_PATH}")
    print("SLURM array should be: --array=1-%d" % len(df))
    print(df["network"].nunique(), "networks x",
          df["noise_type"].nunique(), "noise types x",
          df.groupby(["network", "noise_type"]).size().mean(), "files avg")


if __name__ == "__main__":
    main()
