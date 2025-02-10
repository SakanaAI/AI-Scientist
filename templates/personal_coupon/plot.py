import json
import os
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

# Get experiment folders
experiment_folders = [
    f for f in os.listdir("./") if f.startswith("run") and os.path.isdir(f)
]


def generate_color_palette(n):
    cmap = plt.get_cmap("tab10")
    return [mcolors.rgb2hex(cmap(i)) for i in range(n)]


# Color palette by experiment
colors = generate_color_palette(len(experiment_folders))

plt.figure(figsize=(10, 6))

# Dictionary to store ROI & Lift for each experiment
experiment_metrics = {}

for i, folder in enumerate(experiment_folders):
    coupon_path = Path(folder) / "coupon.csv"
    final_info_path = Path(folder) / "final_info.json"

    # Process coupon distribution plot
    if coupon_path.exists():
        coupon_df = pd.read_csv(coupon_path)
        coupon_counts = (
            coupon_df["coupon_amount"].value_counts().sort_index().reset_index()
        )
        plt.plot(
            coupon_counts["coupon_amount"],
            coupon_counts["count"],
            marker="o",
            color=colors[i],
            label=folder,
        )

    # Process ROI & Lift data
    if final_info_path.exists():
        with open(final_info_path, "r") as f:
            final_info = json.load(f)
            experiment_metrics[folder] = {
                "roi": final_info.get("roi", 0),
                "lift": final_info.get("lift", 0),
            }

# Save coupon distribution plot
plt.xlabel("Coupon amount")
plt.ylabel("Number of coupons")
plt.title("Coupon distribution by amount")
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.savefig("coupon_distribution.png")
plt.close()

# ROI & Lift separate plots
if experiment_metrics:
    labels = list(experiment_metrics.keys())
    roi_values = [experiment_metrics[exp]["roi"] for exp in labels]
    lift_values = [experiment_metrics[exp]["lift"] for exp in labels]

    # ROI Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, roi_values, color="blue", alpha=0.7)
    plt.xlabel("Experiment Run")
    plt.ylabel("ROI")
    plt.title("ROI Comparison Across Experiments")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("roi_comparison.png")
    plt.close()

    # Lift Comparison Plot
    plt.figure(figsize=(10, 6))
    plt.bar(labels, lift_values, color="orange", alpha=0.7)
    plt.xlabel("Experiment Run")
    plt.ylabel("Lift")
    plt.title("Lift Comparison Across Experiments")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("lift_comparison.png")
    plt.close()
