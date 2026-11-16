"""
How many perturbation repeats are needed?

For every curve (network, metric, action, target) the 100 existing repeats are
subsampled: SUBSETS random subsets of each size in SUBSET_SIZES, without
replacement. Each subset gives a mean curve and a failure threshold, which are
compared with the full 100.

The curves are in retention units (value / baseline). The failure threshold
follows scripts/cross_scale_threshholds: the first noise level whose mean is
below 50 % of the baseline, and curves that never fall below it have no
failure. Local AUPRC is measured against its first noise level, like there.

Differences between networks are checked too (empirical vs. each null model,
and empirical networks against each other), with both sides subsampled
independently.

Run build_per_repeat_table.py first. Writes to outputs/repeat_sufficiency/:
  by_curve.csv, by_contrast.csv, summary_by_metric.csv, summary_contrasts.csv

Run from the repository root:
  python scripts/repeat_sufficiency/subsample_repeats.py
"""

import itertools
import warnings

import numpy as np
import pandas as pd

OUT_DIR = "outputs/repeat_sufficiency/"
SUBSET_SIZES = (5, 10, 25, 50)
SUBSETS = 1000
N_REPEATS = 100
RETENTION = 0.5
# Smallest difference (retention units) that counts as a difference to preserve
MIN_DIFFERENCE = 0.02

# Levels where every repeat of infomap / label propagation is undefined stay NaN
warnings.filterwarnings("ignore", message="Mean of empty slice")


def retention_curves(curve, metric, baseline):
    """Matrix repeats x noise levels of retention values."""
    matrix = curve.pivot(index="repeat", columns="noise_level", values="value").to_numpy()
    if metric.startswith("auprc"):
        return matrix / np.nanmean(matrix[:, 0])
    if metric.startswith("jaccard"):
        return matrix
    return matrix / baseline


def failure_step(means):
    """First noise-level index below the threshold; number of levels if none."""
    below = means < RETENTION
    return np.where(below.any(axis=1), below.argmax(axis=1), means.shape[1])


def subset_means(matrix, size, rng):
    order = np.argsort(rng.random((SUBSETS, N_REPEATS)), axis=1)[:, :size]
    return np.nanmean(matrix[order], axis=1)


def ordering(step_a, step_b):
    return np.sign(step_a - step_b)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    df = pd.read_parquet(OUT_DIR + "per_repeat_values.parquet")
    baselines = pd.read_csv(OUT_DIR + "baseline_values.csv").set_index(["metric", "network"])["baseline"]

    full, subsets, rows = {}, {}, []
    for (network, metric, action, target), curve in df.groupby(
        ["network", "metric", "action", "target"], observed=True
    ):
        baseline = None if metric.startswith(("auprc", "jaccard")) else baselines[(metric, network)]
        matrix = retention_curves(curve, metric, baseline)
        key = (network, metric, action, target)
        full[key] = np.nanmean(matrix, axis=0)
        full_step = failure_step(full[key][None, :])[0]
        n_levels = matrix.shape[1]

        for size in SUBSET_SIZES:
            means = subset_means(matrix, size, rng)
            subsets[key + (size,)] = means
            steps = failure_step(means)
            # finite-population correction: subsets of a fixed 100 vary less
            # than independent runs would, most visibly at size 50
            sd = np.nanstd(means, axis=0, ddof=1) / np.sqrt((N_REPEATS - size) / (N_REPEATS - 1))
            rows.append({
                "network": network, "metric": metric, "action": action, "target": target,
                "size": size,
                "curve_sd": np.nanmean(sd),
                "curve_sd_max": np.nanmax(sd),
                "max_abs_dev_p95": np.nanpercentile(np.nanmax(np.abs(means - full[key]), axis=1), 95),
                "fails_in_full": bool(full_step < n_levels),
                "threshold_exact": np.mean(steps == full_step),
                "threshold_within_1": np.mean(np.abs(steps - full_step) <= 1),
            })
    by_curve = pd.DataFrame(rows)
    by_curve.to_csv(OUT_DIR + "by_curve.csv", index=False)

    # --- do the differences we care about survive? ---
    networks = df[["network", "base_network", "model"]].drop_duplicates()
    empirical = sorted(networks.loc[networks["model"] == "empirical", "network"])
    pairs = [("empirical_vs_null", e, f"{e}_{m}") for e in empirical for m in ("er", "conf", "sbm")]
    pairs += [("empirical_vs_empirical", a, b) for a, b in itertools.combinations(empirical, 2)]

    rows = []
    for (metric, action, target) in df[["metric", "action", "target"]].drop_duplicates().itertuples(index=False):
        for kind, a, b in pairs:
            key_a, key_b = (a, metric, action, target), (b, metric, action, target)
            full_diff = full[key_a] - full[key_b]
            step_a, step_b = failure_step(full[key_a][None, :])[0], failure_step(full[key_b][None, :])[0]
            real = np.abs(full_diff) >= MIN_DIFFERENCE

            for size in SUBSET_SIZES:
                means_a, means_b = subsets[key_a + (size,)], subsets[key_b + (size,)]
                diff = means_a - means_b
                sign_agreement = np.nanmean(
                    np.sign(diff[:, real]) == np.sign(full_diff[real])
                ) if real.any() else np.nan
                noise = np.nanstd(diff, axis=0, ddof=1) / np.sqrt((N_REPEATS - size) / (N_REPEATS - 1))
                rows.append({
                    "kind": kind, "a": a, "b": b, "metric": metric, "action": action, "target": target,
                    "size": size,
                    "n_real_levels": int(real.sum()),
                    "sign_agreement": sign_agreement,
                    "noise_over_difference": np.nanmedian(noise[real] / np.abs(full_diff[real])) if real.any() else np.nan,
                    "threshold_order_agreement": np.mean(
                        ordering(failure_step(means_a), failure_step(means_b)) == np.sign(step_a - step_b)
                    ),
                    "thresholds_differ_in_full": bool(step_a != step_b),
                })
    by_contrast = pd.DataFrame(rows)
    by_contrast.to_csv(OUT_DIR + "by_contrast.csv", index=False)

    # --- summaries ---
    def family(metric):
        return metric.split("_")[0]

    by_curve["family"] = by_curve["metric"].map(family)
    summary = by_curve.groupby(["family", "size"]).agg(
        curve_sd_median=("curve_sd", "median"),
        curve_sd_p95=("curve_sd", lambda x: x.quantile(0.95)),
        max_abs_dev_p95=("max_abs_dev_p95", "median"),
    )
    failing = by_curve[by_curve["fails_in_full"]].groupby(["family", "size"]).agg(
        n_failing_curves=("threshold_exact", "size"),
        threshold_exact_mean=("threshold_exact", "mean"),
        threshold_exact_p05=("threshold_exact", lambda x: x.quantile(0.05)),
        threshold_within_1_mean=("threshold_within_1", "mean"),
        threshold_within_1_min=("threshold_within_1", "min"),
    )
    summary = summary.join(failing).round(4)
    summary.to_csv(OUT_DIR + "summary_by_metric.csv")

    by_contrast["family"] = by_contrast["metric"].map(family)
    contrasts = by_contrast.groupby(["kind", "size"]).agg(
        sign_agreement_mean=("sign_agreement", "mean"),
        sign_agreement_p05=("sign_agreement", lambda x: x.quantile(0.05)),
        noise_over_difference_median=("noise_over_difference", "median"),
        noise_over_difference_p95=("noise_over_difference", lambda x: x.quantile(0.95)),
    )
    contrasts = contrasts.join(
        by_contrast[by_contrast["thresholds_differ_in_full"]].groupby(["kind", "size"]).agg(
            threshold_order_agreement_mean=("threshold_order_agreement", "mean"),
            threshold_order_agreement_p05=("threshold_order_agreement", lambda x: x.quantile(0.05)),
        )
    ).round(4)
    contrasts.to_csv(OUT_DIR + "summary_contrasts.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summary.to_string())
    print(contrasts.to_string())
