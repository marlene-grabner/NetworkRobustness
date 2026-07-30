# scripts/submit_communities.py
import os
import subprocess

def get_job_bin(algo, net_key):
    """
    Categorizes a job into one of 4 resource bins based on its 
    computational intensity.
    """
    # Define network sizes based on keys (matching base and null model variations)
    small_nets = ["power", "wiki"]
    
    # Determine network size profile
    is_small_net = any(s in net_key for s in small_nets) or net_key == "test"
    
    if algo == "infomap":
        return "infomap_small" if is_small_net else "infomap_large"
    else:
        return "fast_algos_small" if is_small_net else "fast_algos_large"

def submit_jobs():
    perturbed_root = "./data/perturbed_networks/"
    baseline_npz_dir = "./outputs/local_structure/baselines/"
    out_dir = "./outputs/local_structure/recovery_perturbed/"
    
    algorithms = ["leiden", "infomap", "louvain", "label_propagation"]
    network_keys = [
        "ppi", "astro", "power", "wiki", 
        "ppi_er", "ppi_conf", "ppi_sbm", 
        "astro_er", "astro_conf", "astro_sbm", 
        "power_er", "power_conf", "power_sbm", 
        "wiki_er", "wiki_conf", "wiki_sbm"
    ]

    # Initialize a dictionary to hold jobs for each bin separately
    bins = {
        "fast_algos_small": [],
        "fast_algos_large": [],
        "infomap_small": [],
        "infomap_large": []
    }
    
    for algo in algorithms:
        for net_key in network_keys:
            baseline_npz = os.path.join(baseline_npz_dir, algo, f"{net_key}_{algo}.npz")
            if not os.path.exists(baseline_npz):
                print(f"Warning: Baseline NPZ not found for {net_key} with {algo}. Skipping.")
                continue
                
            net_perturbed_dir = os.path.join(perturbed_root, net_key)
            if not os.path.exists(net_perturbed_dir):
                print(f"Warning: Perturbed directory not found for {net_key}. Skipping.")
                continue
                
            for noise_type in os.listdir(net_perturbed_dir):
                noise_dir = os.path.join(net_perturbed_dir, noise_type)
                if not os.path.isdir(noise_dir):
                    print(f"Warning: Expected directory for noise type {noise_type} in {net_key}. Skipping.")
                    continue
                    
                for fname in os.listdir(noise_dir):
                    if not fname.endswith('.parquet'):
                        print(f"Warning: Non-parquet file {fname} found in {noise_dir}. Skipping.")
                        continue
                        
                    parquet_path = os.path.join(noise_dir, fname)
                    
                    # Create unique output path
                    out_csv_dir = os.path.join(out_dir, algo, net_key, noise_type)
                    os.makedirs(out_csv_dir, exist_ok=True)
                    out_csv = os.path.join(out_csv_dir, fname.replace('.parquet', '.csv'))
                    
                    if not os.path.exists(out_csv):
                        job_bin = get_job_bin(algo, net_key)
                        bins[job_bin].append((parquet_path, baseline_npz, out_csv, algo))

    # Resource Configurations for SLURM based on your benchmarks (adjusted for 6 cores / 10 seeds)
    bin_configs = {
        "fast_algos_small": {"cpus": 6, "mem": "12G",  "time": "01:00:00", "limit": 3},
        "fast_algos_large": {"cpus": 6, "mem": "12G", "time": "02:30:00", "limit": 7},
        "infomap_small":    {"cpus": 6, "mem": "6G",  "time": "02:30:00", "limit": 7},
        "infomap_large":    {"cpus": 8, "mem": "16G", "time": "08:00:00", "limit": 12}
    }

    os.makedirs('slurm_scripts/local_structure/tmp/', exist_ok=True)

    for bin_name, jobs in bins.items():
        total_jobs = len(jobs)
        print(f"Bin [{bin_name}]: {total_jobs} jobs pending.")
        if total_jobs == 0:
            continue



        # 1. Write the specialized task list for this bin
        task_file = f'slurm_scripts/local_structure/tmp/tasks_{bin_name}.txt'
        with open(task_file, 'w') as f:
            for p, b, o, a in jobs:
                f.write(f"{p}\t{b}\t{o}\t{a}\n")

        # 2. Build the tailored SLURM script
        cfg = bin_configs[bin_name]
        slurm_script = f"""#!/bin/bash
#SBATCH --job-name=comm_{bin_name}
#SBATCH --array=1-{total_jobs}%{cfg['limit']}
#SBATCH --cpus-per-task={cfg['cpus']}               
#SBATCH --mem={cfg['mem']}                        
#SBATCH --time={cfg['time']}                 
#SBATCH --output=outputs/logs/comm_{bin_name}_%A_%a.out
#SBATCH --error=outputs/logs/comm_{bin_name}_%A_%a.err

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file})
PARQUET=$(echo "$LINE" | cut -f1)
BASELINE=$(echo "$LINE" | cut -f2)
OUT=$(echo "$LINE" | cut -f3)
ALGO=$(echo "$LINE" | cut -f4)
        
uv run notebooks_general_analysis/local_structure/similarity_perturbation/test_community_pertrubed.py "$PARQUET" "$BASELINE" "$OUT" "$ALGO" --n-jobs {cfg['cpus']}
"""

        script_path = f'slurm_scripts/local_structure/tmp/submit_{bin_name}.sh'
        with open(script_path, 'w') as f:
            f.write(slurm_script)


        # 3. Fire it off to the cluster
        subprocess.run(['sbatch', script_path])
        print(f"-> Submitted {bin_name} array with {total_jobs} tasks.")


if __name__ == "__main__":
    submit_jobs()