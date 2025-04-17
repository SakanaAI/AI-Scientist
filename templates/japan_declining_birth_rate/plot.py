import json
import os
import os.path as osp
import matplotlib.pyplot as plt

# Load policy results
folders = os.listdir("./")
policy_results = {}

for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        file_path = osp.join(folder, "final_info.json")
        if osp.exists(file_path):
            with open(file_path, "r") as f:
                policy_results[folder] = json.load(f)

def plot_policy_impact(results_dict, save_path):
    """Plot predicted birth rate impact for different policies."""
    plt.figure(figsize=(12, 6))
    
    budgets = []
    durations = []
    impacts = []

    for policy_key, data in results_dict.items():
        try:
            policy_tuple = eval(policy_key)  # Convert string key back to tuple
            if not isinstance(policy_tuple, tuple) or len(policy_tuple) != 3:
                print(f"Skipping invalid policy: {policy_key}")
                continue
            
            budget, duration, _ = policy_tuple  # Ignore the third value (effect)
            predicted_impact = data["means"]
            
            budgets.append(budget)
            durations.append(duration)
            impacts.append(predicted_impact)
        except Exception as e:
            print(f"Error processing policy key {policy_key}: {e}")
    
    scatter = plt.scatter(budgets, impacts, c=durations, cmap="coolwarm", alpha=0.7, edgecolors="k")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Policy Duration (years)")

    plt.xlabel("Budget Allocation (Billion Yen)")
    plt.ylabel("Predicted Increase in Birth Rate (per 1000 people)")
    plt.title("Effectiveness of AI-Generated Policies on Birth Rate")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(save_path)
    plt.close()

# Generate plots for each run
for folder, results in policy_results.items():
    print(f"Processing results for {folder}...")
    plot_policy_impact(results, save_path=f"birthrate_impact_{folder}.png")
