import json
import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "seaborn"])
    import seaborn as sns

# LOAD FINAL RESULTS:
datasets = ["SPHERE_Challenge"]
folders = os.listdir("./")
final_results = {}
train_info = {}




for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        all_results = pickle.load(open(osp.join(folder, "all_results.pkl"), "rb"))
        train_info[folder] = all_results

# CREATE LEGEND -- PLEASE FILL IN YOUR RUN NAMES HERE
# Keep the names short, as these will be in the legend.
labels = {
    "run_0": "Baseline",
}

# Use the run key as the default label if not specified
runs = list(final_results.keys())
for run in runs:
    if run not in labels:
        labels[run] = run


# CREATE PLOTS

# Get the list of runs and generate the color palette
runs = list(final_results.keys())


for j, dataset in enumerate(datasets):
    for i, run in enumerate(runs):
        class_names = train_info[run][dataset]['labels']
        cm = train_info[run][dataset]['confusion_matrix']
        fig, ax= plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title(f'Confusion Matrix for {dataset} - {labels[run]}'); 
        ax.xaxis.set_ticklabels(class_names, rotation=320); ax.yaxis.set_ticklabels(class_names, rotation=320);
        # plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{dataset}_{labels[run]}.png", dpi=300)
        plt.show()
