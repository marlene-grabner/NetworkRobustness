"""
Combines the results of calculate_seed_expansion.py:

  outputs/seed_expansion/seed_expansion.parquet
    all rows: one per perturbed network, seed set, algorithm and k
    (~80 million rows; written file by file, never all in memory)
  outputs/seed_expansion/seed_expansion_summary.parquet
    mean and std over the repeats: one row per network, model, noise type,
    level, replicate, seed set, algorithm and k
  outputs/seed_expansion/seed_expansion_baselines.csv
    one row per unperturbed network, seed set, algorithm and k

Run from the repository root:
  uv run python scripts/seed_expansion/expansion/aggregate_seed_expansion.py
"""

import glob

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RESULTS_DIR = "./outputs/seed_expansion/results/"
OUT_DIR = "./outputs/seed_expansion/"

KEYS = [
    "network", "model", "noise_type", "target", "action", "level",
    "replicate", "set_id", "seed_type", "algorithm", "k",
]

if __name__ == "__main__":
    files = sorted(glob.glob(RESULTS_DIR + "*/*/*.parquet"))
    if not files:
        raise SystemExit(f"No results in {RESULTS_DIR}")

    writer, summaries, n_rows = None, [], 0
    for path in files:
        table = pq.read_table(path)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(OUT_DIR + "seed_expansion.parquet", schema)
        writer.write_table(table.cast(schema))
        n_rows += table.num_rows
        # a file is one network, model, noise type and level: its groups are complete
        df = table.to_pandas()
        values = [c for c in df.columns if c not in KEYS + ["repeat"]]
        summary = df.groupby(KEYS)[values].agg(["mean", "std"])
        summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
        summary["n_repeats"] = df.groupby(KEYS).size()
        summaries.append(summary.reset_index())
    writer.close()

    summary = pd.concat(summaries, ignore_index=True)
    pq.write_table(pa.Table.from_pandas(summary, preserve_index=False),
                   OUT_DIR + "seed_expansion_summary.parquet")

    n_files = len(files)
    files = sorted(glob.glob(RESULTS_DIR + "*/*/baseline.csv"))
    baselines = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    baselines.to_csv(OUT_DIR + "seed_expansion_baselines.csv", index=False)

    print(f"{n_rows} perturbed rows from {n_files} files, {len(summary)} summary rows, "
          f"{len(baselines)} baseline rows")
    print(summary.groupby(["network", "model"]).size().unstack().to_string())
