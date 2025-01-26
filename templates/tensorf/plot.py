import json
import os
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def plot_image_grid(image_dirs, datasets, folder):
    num_datasets = len(datasets)
    if num_datasets != len(image_dirs):
        raise ValueError("Number of datasets must match number of image directories")

    total_images = 20
    base_images_per_dataset = total_images // num_datasets
    remainder = total_images % num_datasets

    # Distribute remainder images
    images_per_dataset = [base_images_per_dataset + (1 if i < remainder else 0)
                          for i in range(num_datasets)]

    # Create a 5x4 grid of subplots
    fig, axs = plt.subplots(5, 4, figsize=(15, 12))
    axs = axs.ravel()  # Flatten the 2D array of axes

    current_idx = 0
    for dataset_idx, (image_dir, dataset, num_images) in enumerate(zip(image_dirs, datasets, images_per_dataset)):

        nums = np.linspace(0, 199, num=num_images, dtype=int)
        img_names = [str(num).zfill(3) + ".png" for num in nums]

        # Plot images for this dataset
        for i, img_name in enumerate(img_names):
            img_path = osp.join(image_dir, img_name)
            if osp.exists(img_path):
                img = plt.imread(img_path)
                axs[current_idx].imshow(img)
                axs[current_idx].axis("off")
            current_idx += 1

    # Turn off any remaining empty subplots
    for i in range(current_idx, total_images):
        axs[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"images_all_{folder}.png")
    plt.close()


datasets = ["chair", "drums"]
folders = os.listdir("./")
final_results = {}
results_info = {}

metrics = ["train/loss", "train/PSNR", "train/mse", "test/PSNR", "iters", "train/reg", "train/reg_l1", "train/reg_tv_density", "train/reg_tv_app"]

# Load results and compute metrics
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        results_dict = np.load(osp.join(folder, "all_results.npy"), allow_pickle=True).item()
        run_info = {}
        image_dirs = [results_dict[dataset][0]["imgs"] for dataset in datasets]
        plot_image_grid(image_dirs, datasets, folder)
        for dataset in datasets:
            dset_curr = results_dict[dataset]
            iters = dset_curr[0]["iters"]

            run_info[dataset] = {}

            for metric in metrics:
                # check if metric is empty list 
                losses = [dset_curr[int(i)][metric] for i in dset_curr.keys()]
                if len(losses[0]) == 0:
                    losses = [0] * len(iters)
                    run_info[dataset][metric] = {
                        "iters": iters,
                        "mean": [0] * len(iters),
                        "stderr": [0] * len(iters)
                    }
                    continue

                losses = np.array(losses)
                mean_losses = np.mean(losses, axis=0)
                if len(losses) > 0:
                    sterr_losses = np.std(losses, axis=0) / np.sqrt(len(losses))
                else:
                    sterr_losses = np.zeros_like(mean_losses)

                if metric.startswith("test"):
                    iters_test = [i for i in range(0, len(losses[0]))]
                    run_info[dataset][metric] = {
                        "iters": iters_test,
                        "mean": mean_losses,
                        "stderr": sterr_losses
                    }
                else:
                    run_info[dataset][metric] = {
                        "iters": iters,
                        "mean": mean_losses,
                        "stderr": sterr_losses
                    }
        results_info[folder] = run_info


def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


labels = {
    "run_0": "Baseline",
}
runs = list(labels.keys())
colors = generate_color_palette(len(runs))


# Function to plot metrics
def plot_metric(metric_name, datasets, results_info, runs, colors, labels):
    for dataset in datasets:
        plt.figure(figsize=(10, 6))
        for i, run in enumerate(runs):
            metric_info = results_info[run][dataset].get(metric_name, {})
            iters = metric_info.get("iters", [])
            mean = metric_info.get("mean", [])
            stderr = metric_info.get("stderr", [])
            if iters:
                plt.plot(iters, mean, label=f"{labels[run]} ({metric_name})", color=colors[i])
                plt.fill_between(iters, np.array(mean) - np.array(stderr), np.array(mean) + np.array(stderr),
                                 color=colors[i], alpha=0.2)

        plt.title(f"{metric_name.capitalize()} Across Runs for {dataset} Dataset")
        plt.xlabel("Iteration")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"{metric_name.replace('/', '_')}_{dataset}.png")
        plt.close()


# # Plotting metrics for all datasets
metrics = ["train/PSNR", "train/mse", "test/PSNR", "train/reg", "train/reg_l1", "train/reg_tv_density", "train/reg_tv_app"]
for metric in metrics:
    plot_metric(metric, datasets, results_info, runs, colors, labels)
