import os
import glob
import pandas as pd
import networkx as nx

def standardize_and_verify(input_dir: str, output_dir: str):
    """
    Converts messy TSV/TXT edgelists into strict CSVs while mathematically 
    proving the network structure remains identical.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Grab all files in your input folder (change *.tsv to *.* if they have no extension)
    files = glob.glob(os.path.join(input_dir, "*.tsv"))
    
    print(f"Found {len(files)} files to standardize.\n" + "-"*40)
    
    for file_path in files:
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        new_file_path = os.path.join(output_dir, f"{name}.csv")
        
        print(f"Processing: {filename}...")
        
        # ---------------------------------------------------------
        # 1. THE 'BEFORE' CHECK (Ground Truth via NetworkX)
        # ---------------------------------------------------------
        # NetworkX naturally handles the mixed spaces/tabs
        G_orig = nx.read_edgelist(file_path, comments='#')
        
        orig_nodes = set(G_orig.nodes())
        # We use frozenset for edges because in an undirected graph, 
        # an edge from A->B is topologically identical to B->A.
        orig_edges = set(frozenset(e) for e in G_orig.edges()) 
        
        # ---------------------------------------------------------
        # 2. THE CONVERSION (via Pandas)
        # ---------------------------------------------------------
        # Use our regex fix to safely parse the messy data
        df = pd.read_csv(
            file_path, 
            sep=r'\s+', 
            header=None, 
            usecols=[0, 1], 
            names=['source', 'target'], 
            comment='#',
            dtype=str
        )
        
        # Save as a strict comma-separated CSV with a header row
        df.to_csv(new_file_path, index=False, sep=',', header=False)
        
        # ---------------------------------------------------------
        # 3. THE 'AFTER' CHECK 
        # ---------------------------------------------------------
        # Read the pristine CSV back into a new graph object.
        # We must skip the first line, otherwise NetworkX will treat 
        # the header 'source,target' as an actual node connection!
        with open(new_file_path, 'rb') as f:
            G_new = nx.read_edgelist(f, delimiter=',', comments='#')
            
        new_nodes = set(G_new.nodes())
        new_edges = set(frozenset(e) for e in G_new.edges())
        
        # ---------------------------------------------------------
        # 4. MATHEMATICAL VERIFICATION
        # ---------------------------------------------------------
        nodes_match = (orig_nodes == new_nodes)
        edges_match = (orig_edges == new_edges)

        print(f"  Original nodes: {len(orig_nodes)}, New nodes: {len(new_nodes)}")
        print(f"  Original edges: {len(orig_edges)}, New edges: {len(new_edges)}")
        print(f"Difference in nodes: {orig_nodes.symmetric_difference(new_nodes)}")
        print(f"Difference in edges: {orig_edges.symmetric_difference(new_edges)}")

        if nodes_match and edges_match:
            print(f"  [SUCCESS] Verified! {len(new_nodes)} nodes and {len(new_edges)} edges preserved perfectly.")
        else:
            print(f"  [CRITICAL ERROR] Network structure changed during conversion for {filename}!")
            # Optional: Delete the corrupted file so you don't accidentally use it
            if os.path.exists(new_file_path):
                os.remove(new_file_path)
                print(f"  Deleted corrupted output file: {new_file_path}")

# ==========================================
# Run the pipeline
# ==========================================
input_folder = "./data/baseline_networks/old_tsv_networks/null_models/"   # Update this
output_folder = "./data/baseline_networks/null_models/"   # Update this

standardize_and_verify(input_folder, output_folder)