import re
from pathlib import Path
import pandas as pd

PERTURBED_DIR = Path("./data/perturbed_networks")
MAPPING_CSV = Path("data/network_mapping.csv")

# We will write separate job tables for Short and Long tasks
JOBS_DIR = Path("./slurm_scripts/seed_expansion/expansion/jobs")
SHORT_OUT_PATH = JOBS_DIR / "short_jobs.csv"
LONG_OUT_PATH = JOBS_DIR / "long_jobs.csv"

# Directory where your output metrics files are saved (adjust if your pipeline writes elsewhere)
RESULTS_DIR = Path("./outputs/seed_expansion/expansion/perturbed_metrics")

NOISE_RE = re.compile(r"noise_([0-9p]+)\.parquet$")

# Configure which networks belong to which computational class
SHORT_NETWORKS = {"power", "power_grid", "wikipedia", "wiki", "test"}
LONG_NETWORKS = {"ppi", "collaboration", "astro", "astro_ph"}


def parse_noise_level(filename: str) -> float:
    m = NOISE_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse noise level from filename: {filename}")
    return float(m.group(1).replace("p", "."))


def is_already_completed(network_id: str, perturbation_type: str, modification_type: str, noise_level: float) -> bool:
    """
    Checks if a completed metrics file already exists on disk.
    Supports both .csv and .parquet output extensions under standard layouts.
    """
    # Standard format: e.g., data/results/seed_expansion/astro/hub_target/removal_noise_0.02.csv
    path = RESULTS_DIR / f"{network_id}_{perturbation_type}_{modification_type}_{str(noise_level).replace('.', 'p')}.parquet"

    if path.exists() and path.stat().st_size > 0:
        return True
    
    return False


def classify_job_tier(network_id: str, modification_type: str) -> str:
    """
    Classifies a task as 'short' or 'long'.
    Addition modifications on large/dense graphs are treated as especially heavy.
    """
    net_lower = network_id.lower()
    
    # 1. Force explicitly listed short networks to the short tier
    if any(short_name in net_lower for short_name in SHORT_NETWORKS):
        return "short"
        
    # 2. Force explicitly listed long networks (PPI, Astro/Collaboration) to the long tier
    if any(long_name in net_lower for long_name in LONG_NETWORKS):
        return "long"
        
    # 3. Fallback/Default: Additions are heavier than removals
    return "long" if modification_type == "addition" else "short"


def main():
    files = sorted(PERTURBED_DIR.glob("*/*/*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under {PERTURBED_DIR}/*/*/*.parquet")

    all_rows = []
    completed_count = 0

    for f in files:
        network_id = f.parent.parent.name
        perturbation_type = f.parent.name.removeprefix("perturbed_")

        # Extract modification type
        match_add = re.search(r'(add|edge|insert)', f.name, re.IGNORECASE)
        match_rem = re.search(r'(remov|delet|subtract)', f.name, re.IGNORECASE)
        if match_rem:
            modification_type = "removal"
        elif match_add:
            modification_type = "addition"
        else:
            raise ValueError(f"Could not classify perturbation type for file: {f.name}")

        noise_level = parse_noise_level(f.name)

        # Skip generating a task if the result already exists on disk
        if is_already_completed(network_id, perturbation_type, modification_type, noise_level):
            completed_count += 1
            continue

        all_rows.append({
            "network_id": network_id,
            "network": network_id,
            "perturbation_type": perturbation_type,
            "modification_type": modification_type,
            "noise_level": noise_level,
            "parquet_path": str(f),
        })

    print("==================================================")
    print(f"Total Perturbation Files Scanned: {len(files)}")
    print(f"Already Completed (Skipped):       {completed_count}")
    print(f"Remaining Jobs to Process:         {len(all_rows)}")
    print("==================================================")

    if not all_rows:
        print("🎉 All jobs have already completed! Nothing to write.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_rows)

    # Classify each row into Short vs Long tiers
    df["tier"] = df.apply(lambda r: classify_job_tier(r["network_id"], r["modification_type"]), axis=1)

    # Split the tables
    short_df = df[df["tier"] == "short"].drop(columns=["tier"]).sort_values(["network", "perturbation_type", "noise_level"])
    long_df = df[df["tier"] == "long"].drop(columns=["tier"]).sort_values(["network", "perturbation_type", "noise_level"])

    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    # Write Short Jobs
    if not short_df.empty:
        short_df.to_csv(SHORT_OUT_PATH, index=False)
        print(f"📁 Wrote {len(short_df)} jobs to {SHORT_OUT_PATH}")
        print(f"   💡 SLURM array: --array=1-{len(short_df)}  (Recommended Walltime: 01:30:00)")
    else:
        # Clean up old file if empty
        SHORT_OUT_PATH.unlink(missing_ok=True)

    # Write Long Jobs
    if not long_df.empty:
        long_df.to_csv(LONG_OUT_PATH, index=False)
        print(f"📁 Wrote {len(long_df)} jobs to {LONG_OUT_PATH}")
        print(f"   💡 SLURM array: --array=1-{len(long_df)}  (Recommended Walltime: 08:00:00)")
    else:
        # Clean up old file if empty
        LONG_OUT_PATH.unlink(missing_ok=True)
    print("==================================================")


if __name__ == "__main__":
    main()