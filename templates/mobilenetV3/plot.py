import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp

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
train_losses_per_run = {}
val_losses_per_run = {}
train_iterations_per_run = {}
val_epochs_per_run = {}
test_accuracies = {}  # New dictionary to store test accuracies

# Iterate over runs
for i, run in enumerate(runs):
    data_file = osp.join(run, 'mobilenetv3_cifar10_results.json')
    if osp.exists(data_file):
        with open(data_file, 'r') as f:
            # Load the JSON data
            lines = f.readlines()
            data_str = ''.join(lines)
            data = json.loads(data_str)
            
            # Extract training and validation logs
            train_log_info = data['train_log_info']
            val_log_info = data['val_log_info']
            
            # Extract test accuracy
            test_accuracies[run] = data['final_info']['test_acc']
            
            # Prepare training data
            train_iterations = []
            train_losses = []
            
            # Calculate total batches per epoch
            batches = sorted(set(entry['batch'] for entry in train_log_info))
            if len(batches) > 1:
                batch_step = batches[1] - batches[0]
            else:
                batch_step = 1  # Default to 1 if only one batch
            batches_per_epoch = max(batches) // batch_step + 1
            
            for entry in train_log_info:
                # Calculate a cumulative iteration number
                iteration = entry['epoch'] * batches_per_epoch + entry['batch'] // batch_step
                train_iterations.append(iteration)
                train_losses.append(entry['loss'])
            
            # Store training data
            train_iterations_per_run[run] = train_iterations
            train_losses_per_run[run] = train_losses

            # Prepare validation data
            val_epochs = [entry['epoch'] for entry in val_log_info]
            val_losses = [entry['loss'] for entry in val_log_info]
            
            # Store validation data
            val_epochs_per_run[run] = val_epochs
            val_losses_per_run[run] = val_losses
    else:
        print(f"Data file {data_file} not found for run {run}.")

# Plot 1: Training Loss Across Runs
plt.figure(figsize=(10, 6))
for i, run in enumerate(runs):
    if run in train_iterations_per_run:
        plt.plot(train_iterations_per_run[run], train_losses_per_run[run], 
                 label=labels.get(run, run), color=colors[i])

plt.title('Training Loss Across Runs')
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('train_loss_across_runs.png')
plt.close()

print("Training loss plot saved as 'train_loss_across_runs.png'.")

# Plot 2: Validation Loss Across Runs
plt.figure(figsize=(10, 6))
for i, run in enumerate(runs):
    if run in val_epochs_per_run:
        plt.plot(val_epochs_per_run[run], val_losses_per_run[run], 
                 label=labels.get(run, run), color=colors[i])

plt.title('Validation Loss Across Runs')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()
plt.savefig('val_loss_across_runs.png')
plt.close()

print("Validation loss plot saved as 'val_loss_across_runs.png'.")

# Plot 3: Test Accuracy Across Runs
plt.figure(figsize=(10, 6))
run_names = list(test_accuracies.keys())
accuracies = list(test_accuracies.values())

plt.bar(run_names, accuracies, color=[colors[runs.index(run)] for run in run_names])
plt.title('Test Accuracy Across Runs')
plt.xlabel('Run')
plt.ylabel('Test Accuracy (%)')
plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%

# Add value labels on top of each bar
for i, v in enumerate(accuracies):
    plt.text(i, v, f'{v:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('test_accuracy_across_runs.png')
plt.close()

print("Test accuracy plot saved as 'test_accuracy_across_runs.png'.")