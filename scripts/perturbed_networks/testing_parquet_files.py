#%%
import networkx as nx
import pandas as pd
#%%
file_path = "../../data/perturbed_networks/test_network/perturbed_hub_target/test_targeted_hub_removal_noise_0p95.parquet"
df = pd.read_parquet(file_path)

# If you print it, you'll see your "source", "target", and "repeat" columns
print(df.head())
# %%
import os
os.getcwd()
# %%
# Filter for just the first repeat
df_repeat_0 = df[df["repeat"] == 0]

# Convert that specific slice back into a NetworkX graph
G_0 = nx.from_pandas_edgelist(df_repeat_0, source="source", target="target")

print(f"Network 0 has {G_0.number_of_edges()} edges.")
# %%
all_graphs = []

# Group the large dataframe by the 'repeat' column
for repeat_id, group_df in df.groupby("repeat"):
    
    # Rebuild the NetworkX graph for this specific chunk
    G = nx.from_pandas_edgelist(group_df, source="source", target="target")
    
    # Store it, or do your analysis right here
    all_graphs.append(G)
    
    print(f"Processed repeat {repeat_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

# Now all_graphs[0] is your first network, all_graphs[1] is the second, etc.
# %%
