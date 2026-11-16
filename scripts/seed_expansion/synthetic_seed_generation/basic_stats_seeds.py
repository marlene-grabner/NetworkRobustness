import pandas as pd
import numpy as np
from pathlib import Path


class SeedAnalyzer:
    def __init__(self, csv_file):
        """Load and parse seed data from CSV"""
        self.df = pd.read_csv(csv_file)
        self._process_data()

    def _process_data(self):
        """Parse seed nodes and calculate statistics"""
        self.df["seed_nodes_list"] = (
            self.df["seed_nodes"]
            .str.split(";")
            .apply(lambda x: [int(n) for n in x if n.strip()])
        )
        self.df["num_seeds"] = self.df["seed_nodes_list"].apply(len)
        self.df["degree_deviation"] = (
            self.df["actual_degree"] - self.df["resulting_target_degree"]
        )

    def get_summary_statistics(self):
        """Get aggregated statistics by network and percentile"""
        summary = (
            self.df.groupby(["network_id", "target_percentile"])
            .agg(
                {
                    "resulting_target_degree": "first",
                    "actual_degree": ["mean", "std", "min", "max"],
                    "degree_deviation": ["mean", "max", "min"],
                    "num_seeds": ["mean", "std", "min", "max"],
                    "start_node": "count",  # number of replicates
                }
            )
            .round(3)
        )

        return summary

    def print_table(self):
        """Print simple text table with all stats"""
        data = []

        for (network, percentile), group in self.df.groupby(
            ["network_id", "target_percentile"]
        ):
            target_deg = group["resulting_target_degree"].iloc[0]
            mean_actual = group["actual_degree"].mean()
            std_actual = group["actual_degree"].std()
            max_dev = group["degree_deviation"].abs().max()
            mean_seeds = group["num_seeds"].mean()
            std_seeds = group["num_seeds"].std()
            n_reps = len(group)

            data.append(
                {
                    "Network": network,
                    "Percentile": percentile,
                    "Target k": int(target_deg),
                    "Mean k": f"{mean_actual:.1f}",
                    "Std k": f"{std_actual:.2f}",
                    "Max Dev": int(max_dev) if not np.isnan(max_dev) else 0,
                    "Mean S": f"{mean_seeds:.0f}",
                    "Std S": f"{std_seeds:.2f}",
                    "N_Reps": n_reps,
                }
            )

        result_df = pd.DataFrame(data)
        print("\n" + "=" * 120)
        print("SEED STATISTICS SUMMARY")
        print("=" * 120)
        print(result_df.to_string(index=False))
        print("=" * 120 + "\n")

        return result_df

    def print_latex_table(self):
        """Print simple LaTeX table, keeping network names exactly as provided"""
        print("\n" + "=" * 80)
        print("LATEX TABLE")
        print("=" * 80 + "\n")

        print("\\begin{table}[H]")
        print("  \\centering")
        print("  \\caption{Seed set statistics across networks and quantiles.}")
        print("  \\label{tab:seed_statistics}")
        print("  \\begin{tabular}{lcccccc}")
        print("  \\hline")
        print(
            "  Network & Quantile & $k_t$ & $\\langle k \\rangle$ & Std & $\\Delta k$ & Seeds \\\\"
        )
        print("  \\hline")

        for (network, percentile), group in self.df.groupby(
            ["network_id", "target_percentile"]
        ):
            target_deg = group["resulting_target_degree"].iloc[0]
            mean_actual = group["actual_degree"].mean()
            std_actual = group["actual_degree"].std()
            max_dev = group["degree_deviation"].abs().max()
            mean_seeds = group["num_seeds"].mean()
            n_reps = len(group)

            # Handle NaN std (single replicate)
            std_str = f"{std_actual:.2f}" if not np.isnan(std_actual) else "--"

            print(
                f"  {network.replace('_', ' ')} & {percentile:.2f} & {int(target_deg)} & {mean_actual:.1f} & {std_str} & {int(max_dev)} & {int(mean_seeds)} ({n_reps} reps) \\\\"
            )

        print("  \\hline")
        print("  \\end{tabular}")
        print("\\end{table}\n")


# Usage
if __name__ == "__main__":
    import sys

    csv_file = "outputs/seed_expansion/synthetic_seeds/synthetic_seeds_by_bsf.csv"

    if Path(csv_file).exists():
        analyzer = SeedAnalyzer(csv_file)
        analyzer.print_table()
        analyzer.print_latex_table()
    else:
        print(f"Error: {csv_file} not found")
        print("Usage: python seed_analysis_tool.py <csv_file>")
