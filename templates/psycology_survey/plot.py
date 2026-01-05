"""Plots MDS graphs."""
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# LOAD FINAL RESULTS:
folders = os.listdir("./")
final_results = {}
for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)

def plot_analysis(run: str, name: str, similarity_matrix: list[list[float]]):
    """Plot MDS result for the given similarity matrix."""
    data = np.array(similarity_matrix)
    nmds = MDS(
        n_components=2,
        metric=False,
        max_iter=3000,
        eps=1e-12,
        dissimilarity="precomputed",
        random_state=42,
        n_jobs=1,
        n_init=1,
    )
    data_nmds = nmds.fit_transform(data)

    # TODO: dynamically determine the number of clusters
    # kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    # kmeans.fit(data_nmds)
    # labels = kmeans.labels_

    pca = PCA()
    data_pca = pca.fit_transform(data_nmds)

    fig = plt.figure(1)
    ax = plt.axes([0.0, 0.0, 1.0, 1.0])

    s = 100
    plt.scatter(data_pca[:, 0], data_pca[:, 1], s=s, lw=0, label="NMDS")
    # ADD labels to the axis based on associating the survey response with MDS result.
    # Ex: "left side of x axis has theme A and right side of x axis has theme B, therefore the axis likely means <X dimension>"
    # TODO"
    plt.savefig(f"mds_{name}_{run}.png")
    plt.close(fig)

for run, final_result in final_results.items():
    for key, similarity_matrix in final_result.items():
        # similarity matrix are stored using key with "similarity_matrix" suffix.
        if key.find("similarity_matrix") != -1:
            plot_analysis(run, key, similarity_matrix['matrix'])
