import numpy as np
import warnings
import traceback
import joblib
import re
import json
import os

warnings.filterwarnings("ignore")

import sys
sys.path.insert(1, 'src')
sys.path.insert(2, 'src/samplers')

import check_accuracy as ca

# X = np.loadtxt('sampled_data/laser_parameters.txt')
# y = np.loadtxt('sampled_data/emissivity_curves.txt')

sampler_name = 'random'

X = np.load('sampled_data/input_test_data.npy')[:6]
y = np.load('sampled_data/output_test_data.npy')[:6]

test_data = (X, y)
model = joblib.load('model/model.pkl')

results = ca.error(model, test_data).test_set()

def save_results(sampler_name: str, results: dict, base_dir="."):
    """
    Saves results to: {base_dir}/{sampler_name}/results_{n}/results_{n}.json.
    Automatically increments n based on existing directories.
    """
    
    sampler_path = os.path.join(base_dir, sampler_name)
    os.makedirs(sampler_path, exist_ok=True)
    
    # Find existing results directories and extract iteration numbers
    existing = [
        int(re.search(r"results_(\d+)", d).group(1))
        for d in os.listdir(sampler_path)
        if os.path.isdir(os.path.join(sampler_path, d)) and re.match(r"results_\d+", d)
    ]
    
    if existing:
        next_iter = max(existing) + 1
    else:
        next_iter = 1
    
    # Create new folder for this iteration
    iter_folder = os.path.join(sampler_path, f"results_{next_iter}")
    os.makedirs(iter_folder, exist_ok=True)
    
    # Add iteration number to results
    results["iter"] = next_iter
    
    # Save JSON file inside that folder
    file_path = os.path.join(iter_folder, f"results_{next_iter}.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Results saved to: {file_path}")
    return file_path

save_results(sampler_name, results)


