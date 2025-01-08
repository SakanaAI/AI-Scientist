import json
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Define the datasets you're working with
datasets = ["cifar10"]  # Update as per your datasets

INFO = {"cifar10":
            {"size": 50000}
        }

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baseline",
    # Add more runs here if available
}


# Generate a color palette for the runs
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n - 1)]


# Get the list of runs based on the labels dictionary
runs = list(labels.keys())
colors = generate_color_palette(len(runs) + 1)  # +1 to ensure enough colors

# Initialize dictionaries to store data
results_info = {}  # To store per-run, per-dataset data
final_results = {}  # To store final_info for each run

# Iterate over runs
for i, run in enumerate(runs):
    final_info_file = osp.join(run, 'final_info.json')
    all_results_file = osp.join(run, 'all_results.npy')
    if osp.exists(final_info_file) and osp.exists(all_results_file):
        # Load final_info.json
        with open(final_info_file, 'r') as f:
            final_info = json.load(f)
            final_results[run] = final_info  # Store final_info

        # Load all_results.npy
        results_dict = np.load(all_results_file, allow_pickle=True).item()
        # print(results_dict)

        run_info = {}
        for dataset in datasets:

            iterations_per_epoch = INFO[dataset]["size"] // final_info[dataset]["final_info_dict"]["config"][0]["batch_size"]
            # Initialize lists to collect data across seeds
            all_train_iters = []
            all_train_losses = []
            all_val_iters = []
            all_val_losses = []
            seeds = []  # To keep track of seeds

            # Find all keys corresponding to the dataset and collect data
            keys = [k for k in results_dict.keys()]
            seed_numbers = set(k.split('_')[1] for k in keys if '_' in k)
            # print(dataset, seed_numbers, keys)
            for seed in seed_numbers:
                # Collect train_log_info and val_log_info for each seed
                train_key = f"{dataset}_{seed}_train_log_info"
                val_key = f"{dataset}_{seed}_val_log_info"
                if train_key in results_dict and val_key in results_dict:
                    train_info = results_dict[train_key]
                    val_info = results_dict[val_key]
                    seeds.append(seed)

                    # Extract training data
                    train_iters = [entry['epoch'] * iterations_per_epoch + entry['batch'] for entry in train_info]
                    train_losses = [entry['loss'] for entry in train_info]
                    all_train_iters.append(train_iters)
                    all_train_losses.append(train_losses)

                    # Extract validation data
                    val_iters = [entry['epoch'] * iterations_per_epoch for entry in val_info]
                    val_losses = [entry['loss'] for entry in val_info]
                    all_val_iters.append(val_iters)
                    all_val_losses.append(val_losses)

            # Now compute mean and standard error across seeds

            if all_train_losses:
                # Ensure all lists are the same length for averaging
                min_length = min(len(lst) for lst in all_train_losses)
                train_iters_common = all_train_iters[0][:min_length]
                train_losses_array = np.array([lst[:min_length] for lst in all_train_losses])
                mean_train_losses = np.mean(train_losses_array, axis=0)
                stderr_train_losses = np.std(train_losses_array, axis=0) / np.sqrt(len(train_losses_array))
            else:
                train_iters_common = []
                mean_train_losses = []
                stderr_train_losses = []

            if all_val_losses:
                # Ensure all lists are the same length for averaging
                min_length = min(len(lst) for lst in all_val_losses)
                val_iters_common = all_val_iters[0][:min_length]
                val_losses_array = np.array([lst[:min_length] for lst in all_val_losses])
                mean_val_losses = np.mean(val_losses_array, axis=0)
                stderr_val_losses = np.std(val_losses_array, axis=0) / np.sqrt(len(val_losses_array))
            else:
                val_iters_common = []
                mean_val_losses = []
                stderr_val_losses = []

            # Store in run_info
            run_info[dataset] = {
                'train_iters': train_iters_common,
                'mean_train_losses': mean_train_losses,
                'stderr_train_losses': stderr_train_losses,
                'val_iters': val_iters_common,
                'mean_val_losses': mean_val_losses,
                'stderr_val_losses': stderr_val_losses,
            }

        # Store run_info per run
        results_info[run] = run_info
        # print(run_info)
    else:
        print(f"Data files not found for run {run}.")

# Now, plot the data
# Plot 1: Training Loss Across Runs for each dataset
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        run_data = results_info.get(run, {})
        dataset_data = run_data.get(dataset, {})
        if dataset_data:
            iters = np.array(dataset_data['train_iters'])
            mean_losses = np.array(dataset_data['mean_train_losses'])
            stderr_losses = np.array(dataset_data['stderr_train_losses'])
            label = labels.get(run, run)
            color = colors[i]
            plt.plot(iters, mean_losses, label=label, color=color)
            plt.fill_between(iters, mean_losses - stderr_losses, mean_losses + stderr_losses, color=color, alpha=0.2)

    plt.title(f'Training Loss Across Runs for {dataset}')
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'train_loss_{dataset}_across_runs.png')
    plt.close()
    print(f"Training loss plot for {dataset} saved as 'train_loss_{dataset}_across_runs.png'.")

# Plot 2: Validation Loss Across Runs for each dataset
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    for i, run in enumerate(runs):
        run_data = results_info.get(run, {})
        dataset_data = run_data.get(dataset, {})
        if dataset_data:
            iters = np.array(dataset_data['val_iters'])
            mean_losses = np.array(dataset_data['mean_val_losses'])
            stderr_losses = np.array(dataset_data['stderr_val_losses'])
            label = labels.get(run, run)
            color = colors[i]
            plt.plot(iters, mean_losses, label=label, color=color)
            plt.fill_between(iters, mean_losses - stderr_losses, mean_losses + stderr_losses, color=color, alpha=0.2)

    plt.title(f'Validation Loss Across Runs for {dataset}')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'val_loss_{dataset}_across_runs.png')
    plt.close()
    print(f"Validation loss plot for {dataset} saved as 'val_loss_{dataset}_across_runs.png'.")

# Plot 3: Test Accuracy Across Runs
# Assuming you have test accuracies in final_info.json under each dataset
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    run_names = []
    accuracies = []
    for i, run in enumerate(runs):
        final_info = final_results.get(run, {})
        dataset_info = final_info.get(dataset, {})
        means = dataset_info.get('means', {})
        test_accuracy = means.get('test_acc_mean', None)
        if test_accuracy is not None:
            run_names.append(labels.get(run, run))
            accuracies.append(test_accuracy)

    if run_names and accuracies:
        plt.bar(run_names, accuracies, color=[colors[runs.index(run)] for run in runs if labels.get(run, run) in run_names])
        plt.title(f'Test Accuracy Across Runs for {dataset}')
        plt.xlabel('Run')
        plt.ylabel('Test Accuracy (%)')
        plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%
        # Add value labels on top of each bar
        for i, v in enumerate(accuracies):
            plt.text(i, v, f'{v:.2f}%', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'test_accuracy_{dataset}_across_runs.png')
        plt.close()
        print(f"Test accuracy plot for {dataset} saved as 'test_accuracy_{dataset}_across_runs.png'.")
    else:
        print(f"No test accuracy data available for dataset {dataset}.")
