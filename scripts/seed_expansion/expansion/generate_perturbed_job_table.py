"""
Scans data/perturbed_networks/**/*.parquet and writes one job-table row per
parquet file (= one network x perturbation-type x noise-level combination;
each file internally holds all 100 repeats).

Expected layout, e.g.:
    data/perturbed_networks/astro/perturbed_hub_target/
        ca_AstroPh_targeted_hub_addition_noise_0p1.parquet

    -> network_id        = "astro"            (grandparent folder name)
    -> perturbation_type = "hub_target"        (parent folder name, "perturbed_" prefix stripped)
    -> noise_level       = 0.1                 (parsed from "..._noise_0p1.parquet")

`network_id` (the folder name) must resolve to the same network name used for
the baseline node index / seeds table. If your perturbed folder names don't
match your baseline network names 1:1, add a
data/network_mapping.csv file with columns `network_id,network` and it will
be applied automatically.

Usage:
    uv run scripts/seed_expansion/expansion/generate_perturbed_job_table.py
"""
import re
from pathlib import Path

import pandas as pd

PERTURBED_DIR = Path("./data/perturbed_networks")
MAPPING_CSV = Path("data/network_mapping.csv")
OUT_PATH = Path("./slurm_scripts/seed_expansion/expansion/jobs/perturbed_jobs.csv")

NOISE_RE = re.compile(r"noise_([0-9p]+)\.parquet$")


def parse_noise_level(filename: str) -> float:
    m = NOISE_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse noise level from filename: {filename}")
    return float(m.group(1).replace("p", "."))


def load_mapping() -> dict:
    if MAPPING_CSV.exists():
        df = pd.read_csv(MAPPING_CSV)
        return dict(zip(df["network_id"], df["network"]))
    return {}


def main():
    #mapping = load_mapping()
    files = sorted(PERTURBED_DIR.glob("*/*/*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under {PERTURBED_DIR}/*/*/*.parquet")

    rows = []
    for f in files:
        network_id = f.parent.parent.name
        perturbation_type = f.parent.name.removeprefix("perturbed_")

        # Extract modification type from filename (addition vs removal)
        match_add = re.search(r'(add|edge|insert)', f.name, re.IGNORECASE)
        match_rem = re.search(r'(remov|delet|subtract)', f.name, re.IGNORECASE)
        if match_rem:
            modification_type = "removal"
        elif match_add:
            modification_type = "addition"
        else:
            raise ValueError(f"Could not classify perturbation type for file: {f.name}")

        noise_level = parse_noise_level(f.name)

        rows.append({
            "network_id": network_id,
            "network": network_id,
            "perturbation_type": perturbation_type,
            "modification_type": modification_type,
            "noise_level": noise_level,
            "parquet_path": str(f),
        })

    df = pd.DataFrame(rows).sort_values(["network", "perturbation_type", "noise_level"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} jobs to {OUT_PATH}")
    print("SLURM array should be: --array=1-%d" % len(df))
    print(df["network"].nunique(), "networks x",
          df["perturbation_type"].nunique(), "perturbation types x",
          df.groupby(['network', 'perturbation_type']).size().mean(), "noise levels avg")


if __name__ == "__main__":
    main()