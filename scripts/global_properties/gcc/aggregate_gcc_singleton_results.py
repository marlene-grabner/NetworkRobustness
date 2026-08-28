import os, re, glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

####################################################################################
# Parameters: Output CSV location
output_df_path = "./outputs/global_properties/aggregated_gcc_singletons.csv"
input_csvs_folder = "./outputs/global_properties/gcc_singletons/csvs"
####################################################################################

######################
#   Functions
######################

# ===================================================
# Extract features (network name, action, noise level) from network_id
# ===================================================


def extract_features(network_id):
    """
    Extracts the network name, action, and noise level from the network_id string.
    """
    net_id_str = str(network_id)

    # 1. Extract Network Name (Base + Optional Variation)
    # This looks for your 4 specific base names, optionally followed by _config, _er, or _sbm
    network_pattern = r"(western_us_power_grid|chloe_ppi_lcc_2026_02_23|ca-AstroPH_gcc|wiki-Vote_gcc)(?:_config|_er|_sbm)?"
    network_match = re.search(network_pattern, net_id_str)
    network = network_match.group(0) if network_match else "unknown_network"

    # 2. Extract Action (Addition or Removal)
    if re.search(r"(addition|added)", net_id_str, re.IGNORECASE):
        action = "addition"
    elif re.search(r"(removal|removed)", net_id_str, re.IGNORECASE):
        action = "removal"
    else:
        action = "none"  # Usually applies to baseline

    # 3. Extract Noise Level
    # Looks for "noise_XpY" and converts the "p" to a decimal point
    noise_match = re.search(r"noise_(\d+p\d+|\d+)", net_id_str)
    if noise_match:
        noise_level = float(noise_match.group(1).replace("p", "."))
    else:
        noise_level = 0.0  # Baseline has no noise

    return pd.Series([network, action, noise_level])


def interquartile_range(x):
    """Calculates the IQR, which is the median's equivalent to standard deviation."""
    return x.quantile(0.75) - x.quantile(0.25)


######################
#   Analysis
######################

# ===================================================
# Make dataframe of GCC and singletons per noise level, action, and network
# ===================================================

# --- 1. Read and Combine all CSVs ---
all_files = glob.glob(os.path.join(input_csvs_folder, "*.csv"))
print(f"Found files: {all_files}")

# Read all files and concatenate them into one large DataFrame
df_list = [pd.read_csv(f) for f in all_files]
df_raw = pd.concat(df_list, ignore_index=True)
print(f"Combined DataFrame of all files")
print(df_raw.head())

# --- 2. Extract New Columns ---
# Apply the extraction function to create the new columns
df_raw[["network", "action", "noise_level"]] = df_raw["network_id"].apply(
    extract_features
)

# Rename the perturbation column as requested
df_raw = df_raw.rename(columns={"perturbation_method": "noise_type"})

print("After reshaping")
print(df_raw.head())

# --- 3. Aggregate the Repeats ---
# Group by our newly extracted identifiers
grouped = df_raw.groupby(["network", "noise_type", "action", "noise_level"])

# Calculate mean, median, std, and IQR for the numeric columns
df_agg = grouped.agg(
    {
        "num_singletons": ["mean", "median", "std", interquartile_range],
        "gcc": ["mean", "median", "std", interquartile_range],
    }
).reset_index()

print("After calculation of mean, median, std, and IQR")
print(df_agg.head())

# --- 4. Clean Up Column Names ---
# The aggregation creates MultiIndex columns (e.g., ('gcc', 'mean')). Let's flatten them.
df_agg.columns = ["_".join(col).strip("_") for col in df_agg.columns.values]

# Rename the custom IQR function columns for clarity
df_agg = df_agg.rename(
    columns={
        "num_singletons_interquartile_range": "num_singletons_iqr",
        "gcc_interquartile_range": "gcc_iqr",
    }
)

print("After cleaning up column names")
print(df_agg.head())

print(f"Saving the aggregated dataframe as CSV to: {output_df_path}")
df_agg.to_csv(output_df_path, index=False)
