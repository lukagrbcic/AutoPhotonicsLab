import os
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# plt.style.use("seaborn-v0_8-whitegrid")  # scientific clean look
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "mathtext.fontset": "stix",
    "font.family": "serif"
})


def aggregate_results(sampler_name: str, base_dir="."):
    """
    Reads all result JSON files for a given sampler_name and returns
    an aggregated dictionary where each key contains a list of its values
    for all iterations.
    """
    sampler_path = os.path.join(base_dir, sampler_name)
    if not os.path.exists(sampler_path):
        raise FileNotFoundError(f"No results found for sampler: {sampler_name}")

    # Collect file paths in iteration order
    result_files = []
    for folder in os.listdir(sampler_path):
        match = re.match(r"results_(\d+)", folder)
        if match:
            iter_num = int(match.group(1))
            json_path = os.path.join(sampler_path, folder, f"results_{iter_num}.json")
            if os.path.exists(json_path):
                result_files.append((iter_num, json_path))

    # Sort by iteration number
    result_files.sort(key=lambda x: x[0])

    # Aggregate results
    aggregated = defaultdict(list)
    for iter_num, file_path in result_files:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Append values (excluding 'iter' itself)
        for key, value in data.items():
            if key != "iter":
                aggregated[key].append(value)

    # Convert to regular dict and add sampler_name
    aggregated = dict(aggregated)
    aggregated["sampler_name"] = sampler_name

    return aggregated

# Example usage:
agg = aggregate_results("model_uncertainty")
print(json.dumps(agg, indent=4))

def plot_metrics(samplers, metrics, batch_size, max_size, base_dir="."):
    """
    Plots metrics for multiple samplers in a scientific style.

    samplers   - list of sampler names (folders)
    metrics    - list of metric names to plot
    batch_size - step size for x-axis
    max_size   - maximum sample size for x-axis
    base_dir   - directory containing sampler folders
    """

    # Metric name overrides for LaTeX formatting
    metric_labels = {
        "r2": r"$R^2$",
        "mape": r"MAPE (\%)",
        "rmse": r"RMSE",
        "range_nrmse_": r"Range NRMSE",
        "std_nrmse_": r"Std NRMSE",
        "max_rmse_": r"Max RMSE",
        "max_range_nrmse": r"Max Range NRMSE",
        "nmax_ae": r"N Max AE",
        "paper_rmse": r"Paper RMSE",
        "paper_rmse_max": r"Paper RMSE Max"
    }

    agg_list = [aggregate_results(s, base_dir) for s in samplers]
    x_vals = list(range(batch_size, batch_size * (max_size // batch_size + 1), batch_size))

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        for data in agg_list:
            if metric not in data:
                print(f"⚠ Metric '{metric}' not found in sampler '{data['sampler_name']}'")
                continue
            plt.plot(
                x_vals[:len(data[metric])],
                data[metric],
                marker='o',
                markersize=6,
                linewidth=2,
                label=data["sampler_name"]
            )
        
        plt.xlabel("Batch size", fontsize=16)
        plt.ylabel(metric_labels.get(metric, metric), fontsize=16)
        plt.title(f"{metric_labels.get(metric, metric)} vs Batch Size", fontsize=18)
        plt.legend()
        # plt.grid(True, which='both', linestyle='--', linewidth=0.7)
        # plt.minorticks_on()
        plt.tight_layout()
        plt.show()


# -------------------
# Example usage
# -------------------
metrics_to_plot = ["r2", "mape"]
batch_size = 10
max_size = 60
samplers = ["model_uncertainty", "random"]

plot_metrics(samplers, metrics_to_plot, batch_size, max_size)