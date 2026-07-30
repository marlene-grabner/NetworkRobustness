import NoiseEffect as na
from pathlib import Path
import sys

# Extract the key for network to be analyzed
baseline_key = sys.argv[1]

noise_levels_removed = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
noise_levels_added = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 1.50, 2.00]

############################################################
# File paths
############################################################

# Paths of unperturbed baseline networks
#########################################
baseline_files = {
    "chloe_ppi_lcc_2026_02_23": "./data/baseline_networks/chloe_ppi_lcc_2026_02_23.tsv",
    "chloe_ppi_lcc_2026_02_23_er": "./data/baseline_networks/null_models/chloe_ppi_erdos_renyi.tsv",
    "chloe_ppi_lcc_2026_02_23_config": "./data/baseline_networks/null_models/chloe_ppi_configuration_model.tsv",
    "chloe_ppi_lcc_2026_02_23_sbm": "./data/baseline_networks/null_models/chloe_ppi_sbm.tsv",
    "western_us_power_grid": "./data/baseline_networks/western_us_power_grid.tsv",
    "western_us_power_grid_er": "./data/baseline_networks/null_models/western_us_power_grid_erdos_renyi.tsv",
    "western_us_power_grid_config": "./data/baseline_networks/null_models/western_us_power_grid_configuration_model.tsv",
    "western_us_power_grid_sbm": "./data/baseline_networks/null_models/western_us_power_grid_sbm.tsv",
    "ca-AstroPH_gcc": "./data/baseline_networks/ca-AstroPh_gcc.tsv",
    "ca-AstroPH_gcc_er": "./data/baseline_networks/null_models/ca-AstroPh_erdos_renyi.tsv",
    "ca-AstroPH_gcc_config": "./data/baseline_networks/null_models/ca-AstroPh_configuration_model.tsv",
    "ca-AstroPH_gcc_sbm": "./data/baseline_networks/null_models/ca-AstroPh_sbm.tsv",
    "wiki-Vote_gcc": "./data/baseline_networks/wiki-Vote_gcc.tsv",
    "wiki-Vote_gcc_er": "./data/baseline_networks/null_models/wiki-Vote_erdos_renyi.tsv",
    "wiki-Vote_gcc_config": "./data/baseline_networks/null_models/wiki-Vote_configuration_model.tsv",
    "wiki-Vote_gcc_sbm": "./data/baseline_networks/null_models/wiki-Vote_sbm.tsv"
}

# Paths of output folders for random perturbations
#########################################
output_folders_random = {
    "chloe_ppi_lcc_2026_02_23": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23/perturbed_random_target",
    "chloe_ppi_lcc_2026_02_23_er": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23_erdos_renyi/perturbed_random_target",
    "chloe_ppi_lcc_2026_02_23_config": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23_configuration_model/perturbed_random_target",
    "chloe_ppi_lcc_2026_02_23_sbm": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23_sbm/perturbed_random_target",
    "western_us_power_grid": "./data/perturbed_networks/western_us_power_grid/perturbed_random_target",
    "western_us_power_grid_er": "./data/perturbed_networks/western_us_power_grid_erdos_renyi/perturbed_random_target",
    "western_us_power_grid_config": "./data/perturbed_networks/western_us_power_grid_configuration_model/perturbed_random_target",
    "western_us_power_grid_sbm": "./data/perturbed_networks/western_us_power_grid_sbm/perturbed_random_target",
    "ca-AstroPH_gcc": "./data/perturbed_networks/ca-AstroPh_gcc/perturbed_random_target",
    "ca-AstroPH_gcc_er": "./data/perturbed_networks/ca-AstroPh_erdos_renyi/perturbed_random_target",
    "ca-AstroPH_gcc_config": "./data/perturbed_networks/ca-AstroPh_configuration_model/perturbed_random_target",
    "ca-AstroPH_gcc_sbm": "./data/perturbed_networks/ca-AstroPh_sbm/perturbed_random_target",
    "wiki-Vote_gcc": "./data/perturbed_networks/wiki-Vote_gcc/perturbed_random_target",
    "wiki-Vote_gcc_er": "./data/perturbed_networks/wiki-Vote_erdos_renyi/perturbed_random_target",
    "wiki-Vote_gcc_config": "./data/perturbed_networks/wiki-Vote_configuration_model/perturbed_random_target",
    "wiki-Vote_gcc_sbm": "./data/perturbed_networks/wiki-Vote_sbm/perturbed_random_target"
}

# Paths of of output folders for hub-targeted perturbations
#########################################
output_folders_hub_targeted = {
    "chloe_ppi_lcc_2026_02_23": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23/perturbed_hub_target",
    "chloe_ppi_lcc_2026_02_23_er": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23_erdos_renyi/perturbed_hub_target",
    "chloe_ppi_lcc_2026_02_23_config": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23_configuration_model/perturbed_hub_target",
    "chloe_ppi_lcc_2026_02_23_sbm": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23_sbm/perturbed_hub_target",
    "western_us_power_grid": "./data/perturbed_networks/western_us_power_grid/perturbed_hub_target",
    "western_us_power_grid_er": "./data/perturbed_networks/western_us_power_grid_erdos_renyi/perturbed_hub_target",
    "western_us_power_grid_config": "./data/perturbed_networks/western_us_power_grid_configuration_model/perturbed_hub_target",
    "western_us_power_grid_sbm": "./data/perturbed_networks/western_us_power_grid_sbm/perturbed_hub_target",
    "ca-AstroPH_gcc": "./data/perturbed_networks/ca-AstroPh_gcc/perturbed_hub_target",
    "ca-AstroPH_gcc_er": "./data/perturbed_networks/ca-AstroPh_erdos_renyi/perturbed_hub_target",
    "ca-AstroPH_gcc_config": "./data/perturbed_networks/ca-AstroPh_configuration_model/perturbed_hub_target",
    "ca-AstroPH_gcc_sbm": "./data/perturbed_networks/ca-AstroPh_sbm/perturbed_hub_target",
    "wiki-Vote_gcc": "./data/perturbed_networks/wiki-Vote_gcc/perturbed_hub_target",
    "wiki-Vote_gcc_er": "./data/perturbed_networks/wiki-Vote_erdos_renyi/perturbed_hub_target",
    "wiki-Vote_gcc_config": "./data/perturbed_networks/wiki-Vote_configuration_model/perturbed_hub_target",
    "wiki-Vote_gcc_sbm": "./data/perturbed_networks/wiki-Vote_sbm/perturbed_hub_target"
}

# Paths of of output folders for periphery-targeted perturbations
#########################################
output_folders_periphery_targeted = {
    "chloe_ppi_lcc_2026_02_23": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23/perturbed_periphery_target",
    "chloe_ppi_lcc_2026_02_23_er": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23_erdos_renyi/perturbed_periphery_target",
    "chloe_ppi_lcc_2026_02_23_config": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23_configuration_model/perturbed_periphery_target",
    "chloe_ppi_lcc_2026_02_23_sbm": "./data/perturbed_networks/chloe_ppi_lcc_2026_02_23_sbm/perturbed_periphery_target",
    "western_us_power_grid": "./data/perturbed_networks/western_us_power_grid/perturbed_periphery_target",
    "western_us_power_grid_er": "./data/perturbed_networks/western_us_power_grid_erdos_renyi/perturbed_periphery_target",
    "western_us_power_grid_config": "./data/perturbed_networks/western_us_power_grid_configuration_model/perturbed_periphery_target",
    "western_us_power_grid_sbm": "./data/perturbed_networks/western_us_power_grid_sbm/perturbed_periphery_target",
    "ca-AstroPH_gcc": "./data/perturbed_networks/ca-AstroPh_gcc/perturbed_periphery_target",
    "ca-AstroPH_gcc_er": "./data/perturbed_networks/ca-AstroPh_erdos_renyi/perturbed_periphery_target",
    "ca-AstroPH_gcc_config": "./data/perturbed_networks/ca-AstroPh_configuration_model/perturbed_periphery_target",
    "ca-AstroPH_gcc_sbm": "./data/perturbed_networks/ca-AstroPh_sbm/perturbed_periphery_target",
    "wiki-Vote_gcc": "./data/perturbed_networks/wiki-Vote_gcc/perturbed_periphery_target",
    "wiki-Vote_gcc_er": "./data/perturbed_networks/wiki-Vote_erdos_renyi/perturbed_periphery_target",
    "wiki-Vote_gcc_config": "./data/perturbed_networks/wiki-Vote_configuration_model/perturbed_periphery_target",
    "wiki-Vote_gcc_sbm": "./data/perturbed_networks/wiki-Vote_sbm/perturbed_periphery_target"
}


############################################################
# Determine the parameters
############################################################

PERTURBATIONS = {
    "random": {
        "folder": output_folders_random[baseline_key],
        "noise_types": ["added_edges", "removed_edges"]
    },
    "hub": {
        "folder": output_folders_hub_targeted[baseline_key],
        "noise_types": ["targeted_hub_addition", "targeted_hub_removal"]
    },
    "periphery": {
        "folder": output_folders_periphery_targeted[baseline_key],
        "noise_types": ["targeted_periphery_addition", "targeted_periphery_removal"]
    }
}

############################################################
# Run the code to generate the perturbed networks
############################################################

for perturbation_type in PERTURBATIONS.keys():
    # Create the folder if it does not exist
    if not Path(PERTURBATIONS[perturbation_type]["folder"]).exists():
        Path(PERTURBATIONS[perturbation_type]["folder"]).mkdir(parents=True, exist_ok=True)
    # Generate the perturbed networks
    # Addition
    na.generateNoiseNetworksFromBaseline(
        path_to_edgelist=baseline_files[baseline_key],
        folder_to_save_perturbed=PERTURBATIONS[perturbation_type]["folder"],
        noise_levels=noise_levels_added,
        noise_types=[PERTURBATIONS[perturbation_type]["noise_types"][0]],
        num_repeats_per_noise_level=100,
        network_name=baseline_key,
    )

    # Removal
    na.generateNoiseNetworksFromBaseline(
        path_to_edgelist=baseline_files[baseline_key],
        folder_to_save_perturbed=PERTURBATIONS[perturbation_type]["folder"],
        noise_levels=noise_levels_removed,
        noise_types=[PERTURBATIONS[perturbation_type]["noise_types"][1]],
        num_repeats_per_noise_level=100,
        network_name=baseline_key,
    )
