# NoiseInNetworks

A research project investigating **how structural noise (edge additions and deletions) affects networks across three scales:** global, mesoscale (community structure), and local (seed set expansion).

## Summary

Real-world network data is inherently noisy — edges may be missing or spurious. This project systematically studies the robustness of network analysis methods under controlled edge perturbations. Four empirical networks (protein-protein interaction, astrophysics collaboration, US power grid, Wikipedia voting) and their null models (Erdos-Renyi, configuration model, and stochastic block model) are perturbed at multiple noise levels using three strategies (random, hub-targeted, periphery-targeted). The effect is measured at three structural scales:

- **Global scale:** Giant connected component (GCC) fraction and algebraic connectivity (Fiedler eigenvalue).
- **Mesoscale:** Community detection stability using Leiden, Louvain, Infomap, and Label Propagation.
- **Local scale:** Seed set expansion algorithms (DIAMOnD, First Neighbors, Random Walk with Restart).

## Folder Structure

```
.
├── pyproject.toml            # Project metadata and Python dependencies
├── uv.lock                   # Locked dependency versions
├── test.py                   # Ad-hoc test/scratch script
├── README.md                 # This file
│
├── src/NoiseEffect/          # Core Python library
│   ├── NoiseNetworks/        # Generate perturbed networks from baselines
│   ├── GlobalProperties/     # GCC, singleton, and algebraic connectivity computation
│   ├── CommunityDetection/   # Leiden, Louvain, Infomap, Label Propagation wrappers
│   ├── TopologicalProperties/# Network profiling and degree distribution plotting
│   ├── SeedExpansion/        # Seed set expansion algorithms (DIAMOnD, FN, RWR)
│   ├── ModuleRecovery/       # Legacy module detection benchmarking (sanity checks only)
│   ├── NoisePipeline/        # Legacy / superseded pipeline
│   ├── CompareModules/       # Legacy
│   ├── SeedStrucutralMetrics/# Legacy
│   └── utils/                # Shared utilities
│
├── scripts/                  # Executable analysis pipelines
│   ├── models/               # Generate null models and baseline network profiles
│   ├── perturbed_networks/   # Generate perturbed edge sets (.parquet)
│   ├── global_properties/    # GCC, Fiedler value calculation, plotting
│   ├── local_structure/      # Community detection baselines, similarity under noise, stability
│   ├── seed_expansion/       # Seed expansion baselines, perturbed metrics, sanity checks
│   └── cross_scale_threshholds/ # Cross-scale failure threshold heatmaps
│
├── data/                     # Network data
│   ├── baseline_networks/    # Empirical networks (.csv/.tsv) + null models
│   └── perturbed_networks/   # Perturbed networks by network/noise-type (.parquet)
│
├── outputs/                  # Generated results (CSVs, figures, aggregated metrics)
├── tests/                    # Unit tests (pytest)
├── slurm_scripts/            # HPC job submission scripts (SLURM)
└── logs/                     # SLURM job logs
```

## Networks

The project uses four real-world networks and three null model variants per network (16 datasets total):

| Network | Type | Nodes (approx.) |
|---|---|---|
| **PPI** (Protein-Protein Interaction) | Biological | ~5,100 |
| **Astro** (Astrophysics Collaboration) | Collaboration | ~18,700 |
| **Power Grid** (Western US Power Grid) | Infrastructure | ~4,900 |
| **Wiki** (Wikipedia Administrator Voting) | Social | ~7,000 |

Each empirical network has three null models:
- **Erdos-Renyi (ER)** — same node count and edge count, random topology
- **Configuration Model (Conf)** — degree-preserving random graph
- **Stochastic Block Model (SBM)** — preserves inferred block structure

### Perturbation Types

Edges are modified using three targeting strategies:

| Strategy | Effect |
|---|---|
| **Random** | Adds/removes edges uniformly at random |
| **Hub-targeted** | Targets edges incident to high-degree nodes |
| **Periphery-targeted** | Targets edges incident to low-degree nodes |

Noise levels range from 0.02–0.50 (fraction removed) and 0.05–2.00 (fraction added). Each (network, noise_type, noise_level) combination is repeated 100 times with different random seeds.

## Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- For SBM null models: `graph-tool` (requires a separate environment, see `scripts/models/sbm_model_with_graphtool_env.py`)

### Dependencies

Core dependencies (from `pyproject.toml`):
- `networkx`, `igraph`, `leidenalg` — graph handling and community detection
- `numpy`, `pandas`, `scipy`, `scikit-learn` — numerical computation
- `matplotlib`, `seaborn` — plotting
- `infomap` — Infomap community detection
- `pyarrow` — Parquet file I/O
- `tqdm` — progress bars

## Setup and Execution

### 1. Create environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2. Run analysis scripts

All scripts should be executed from the project root directory.

#### Generate null models

```bash
uv run python scripts/models/generate_null_models.py
uv run python scripts/models/properties_baseline_networks.py
```

#### Generate perturbed networks

```bash
uv run python scripts/perturbed_networks/generating_perturbed_networks.py <baseline_key>
```

The `<baseline_key>` corresponds to entries in the script, e.g. `ca-AstroPH_gcc`, `chloe_ppi_lcc_2026_02_23`, `western_us_power_grid`, `wiki-Vote_gcc`.

#### Compute global properties (GCC, singletons)

```bash
uv run python scripts/global_properties/giant_connected_component.py
```

#### Compute algebraic connectivity (Fiedler value)

```bash
uv run python scripts/global_properties/algebraic_connectivity/calculation/fiedler_value_baseline_graphs.py
uv run python scripts/global_properties/algebraic_connectivity/calculation/fiedler_value.py
```

#### Community detection baseline and perturbation analysis

```bash
uv run python scripts/local_structure/baselines.py
uv run python scripts/local_structure/community_similarity_under_perturbation.py
```

#### Seed expansion analysis

```bash
uv run python scripts/seed_expansion/expansion/run_baseline_task.py --help
uv run python scripts/seed_expansion/expansion/run_perturbed_task.py --help
uv run python scripts/seed_expansion/synthetic_seed_generation/synthetic_seed_generation.py --help
```

#### Cross-scale failure threshold plots

```bash
uv run python scripts/cross_scale_threshholds/plot_cross_scale_failure_threshhold.py
uv run python scripts/cross_scale_threshholds/plot_corss_scale_failure_thresshold_linear.py
```

### 3. HPC cluster execution

SLURM scripts are provided in `slurm_scripts/` for batch processing on HPC clusters. Example:

```bash
sbatch slurm_scripts/perturbed_networks/generating_perturbed_networks.slurm
sbatch slurm_scripts/global_properties/giant_connected_component.slurm
```

### 4. Running tests

```bash
uv run pytest tests/
```

## Core Modules (src/NoiseEffect/)

### NoiseNetworks
Generates perturbed versions of a baseline network by randomly or targeted adding/removing edges. Supports control over noise type (random, hub-targeted, periphery-targeted), noise levels, and multiple repeats. Results are saved as `.parquet` files with columns `(source, target, repeat)`.

### GlobalProperties
- `calculate_singletons_and_gcc()` — For each perturbed repeat, computes the number of isolated nodes (singletons) and the fraction of nodes in the giant connected component relative to the baseline.
- `fiedler_on_gcc()` — Computes the algebraic connectivity (Fiedler eigenvalue) of the Laplacian of the giant connected component.

### CommunityDetection
Wraps four community detection algorithms from igraph: **Leiden**, **Louvain**, **Infomap**, and **Label Propagation**. Provides:
- Baseline partition generation across multiple random seeds (saved as `.npz` matrices)
- Comparison of perturbed partitions to baseline using Adjusted Rand Index (ARI)
- Partition stability benchmarking

### TopologicalProperties
- `get_network_profile()` — Computes basic network statistics (node count, edge count, average degree, density).
- `plot_degree_distribution()` — Plots degree distribution with optional log-binning and trend fitting.

### SeedExpansion
Implements seed set expansion algorithms that start from a set of seed nodes and expand outward to recover a local module:
- **DIAMOnD** — Diamond algorithm for module discovery
- **First Neighbors** — Direct neighborhood expansion
- **Random Walk with Restart** — RWR-based ranking with row or symmetric normalization

Includes utilities for synthetic seed generation (BFS-based), ranking comparison, and I/O handling for seed tables and node indices.

## Outputs

| Directory | Contents |
|---|---|
| `outputs/global_properties/` | Aggregated GCC/singleton CSVs, Fiedler value tables, figures |
| `outputs/local_structure/` | Baseline `.npz` partitions, perturbed recovery results, aggregated CSVs, figures |
| `outputs/seed_expansion/` | Baseline rankings, perturbed metrics, synthetic seed tables, figures |
| `outputs/cross_scale_threshholds/` | Cross-scale failure threshold heatmaps |

## Acknowledgments

This project was developed at the [Menche Lab](https://www.menchelab.com/), University of Vienna.