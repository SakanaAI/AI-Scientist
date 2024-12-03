import glob
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datasets = ["south-building"]
all_run_folders = glob.glob("run_*")

all_final_info = {}

for run in all_run_folders:
    with open(os.path.join(run, "final_info.json"), "r") as f:
        all_final_info[run] = json.load(f)

# Plot performance across runs
for dataset in datasets:
    for metric in ["l1", "psnr"]:
        plt.figure(figsize=(10, 6))
        current_result = []
        for run in all_run_folders:
            for split in ["train", "test"]:
                performance = all_final_info[run][dataset]["means"][f"{split}_{metric}"]
                current_result.append([split, run, performance])
        
        df = pd.DataFrame(
            data=current_result,
            columns=["split", "run", "performance"]
        )
        
        plt.title(f"Train and Test {metric} over all runs")
        plt.xlabel("Runs")
        plt.ylabel(metric)
        sns.barplot(df, x="run", y="performance", hue="split")
        plt.savefig(f"{dataset}_{metric}.png")
        plt.close()


