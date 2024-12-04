import glob
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

datasets = ["south-building"]
all_run_folders = glob.glob("run_*")

all_final_info = {}

for run in all_run_folders:
    with open(os.path.join(run, "final_info.json"), "r") as f:
        all_final_info[run] = json.load(f)

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baselines",
}

# Plot performance across runs
for dataset in datasets:
    for metric in ["l1", "psnr"]:
        plt.figure(figsize=(10, 6))
        current_result = []
        for run in all_run_folders:
            for split in ["train", "test"]:
                performance = all_final_info[run][dataset]["means"][f"{split}_{metric}"]
                current_result.append([split, labels[run], performance])

        df = pd.DataFrame(
            data=current_result,
            columns=["split", "run", "performance"]
        )

        plt.title(f"Train and Test {metric} over all runs")
        plt.xlabel("Runs")
        plt.ylabel(metric)
        sns.barplot(df, x="run", y="performance", hue="split")
        plt.xticks(rotation=45)
        plt.savefig(f"{dataset}_{metric}.png")
        plt.close()

# Plot comparison of ground truth image with rendered images from each run
for dataset in datasets:
    subplot_width = 10 * (len(all_run_folders) + 1)
    subplot_height = subplot_width // 2
    fig, ax = plt.subplots(1, len(all_run_folders) + 1, figsize=(subplot_width, subplot_height))
    with Image.open(os.path.join("run_0", f"{dataset}_ground_truth_image.png")) as image:
        image_array = np.array(image)
        ax[0].imshow(image_array)
        ax[0].set_title("Ground Truth")
        ax[0].axis('off')
    for idx, run in enumerate(all_run_folders):
        with Image.open(os.path.join(run, f"{dataset}_rendered_image.png")) as image:
            image_array = np.array(image)
            ax[idx + 1].imshow(image_array)
            ax[idx + 1].set_title(labels[run])
            ax[idx + 1].axis('off')


    plt.savefig(f"{dataset}_images_comparison.png")
    plt.close()

