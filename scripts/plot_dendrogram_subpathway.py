"""
Regenerate dendrogram colored by sub-pathway (MMR/NER/HR/BER/TSG).
Run from: results/evo2_function/
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
import urllib.request
import time

# Paths — adjust if running from elsewhere
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS = os.path.join(_root, "templates/evo2_function/run_0/data/embeddings_evo2_7b.npy")
DATASET    = os.path.join(_root, "templates/evo2_function/run_0/data/dataset.json")

PATHWAY_IDS = {
    "MMR": "hsa03430",
    "NER": "hsa03420",
    "HR":  "hsa03440",
    "BER":  "hsa03410",
}

def fetch_kegg_symbols(pathway_id):
    url = f"https://rest.kegg.jp/link/hsa/{pathway_id}"
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read().decode()
    kegg_ids = [line.split("\t")[1].strip() for line in text.strip().split("\n") if line]
    symbols = []
    for kid in kegg_ids:
        try:
            with urllib.request.urlopen(f"https://rest.kegg.jp/list/{kid}", timeout=15) as r:
                t = r.read().decode().strip()
            if t:
                sym = t.split("\t")[1].split(",")[0].split(";")[0].strip().upper()
                symbols.append(sym)
            time.sleep(0.15)
        except Exception:
            continue
    return set(symbols)

print("Loading embeddings and dataset...")
embeddings = np.load(EMBEDDINGS)
with open(DATASET) as f:
    dataset = json.load(f)

symbols = [d["symbol"].upper() for d in dataset]
labels  = [d["label_name"] for d in dataset]

print("Fetching sub-pathway gene lists from KEGG...")
pathway_genes = {}
for name, pid in PATHWAY_IDS.items():
    print(f"  Fetching {name} ({pid})...")
    pathway_genes[name] = fetch_kegg_symbols(pid)

# Assign sub-pathway label to each gene
def get_subpathway(symbol, label_name):
    if label_name == "tumor_suppressor":
        return "TSG"
    for name, genes in pathway_genes.items():
        if symbol in genes:
            return name
    return "DNA_repair_other"

subpathways = [get_subpathway(s, l) for s, l in zip(symbols, labels)]

# Color map
COLOR_MAP = {
    "MMR":              "#e41a1c",  # red
    "NER":              "#377eb8",  # blue
    "HR":               "#4daf4a",  # green
    "BER":              "#ff7f00",  # orange
    "DNA_repair_other": "#999999",  # grey
    "TSG":              "#984ea3",  # purple
}

leaf_colors = [COLOR_MAP[sp] for sp in subpathways]

print("Computing hierarchical clustering...")
dists   = pdist(embeddings, metric="cosine")
Z       = linkage(dists, method="ward")
coph, _ = cophenet(Z, dists)
print(f"Cophenetic correlation: {coph:.4f}")

# Map leaf colors into dendrogram
fig, ax = plt.subplots(figsize=(28, 8))

ddata = dendrogram(
    Z,
    labels=symbols,
    ax=ax,
    leaf_rotation=90,
    leaf_font_size=4,
    color_threshold=0,
    above_threshold_color="#aaaaaa",
    no_plot=True,
)

# Redraw with custom leaf colors
dendrogram(
    Z,
    labels=symbols,
    ax=ax,
    leaf_rotation=90,
    leaf_font_size=4,
    color_threshold=0,
    above_threshold_color="#cccccc",
)

# Color the x-tick labels by sub-pathway
order = ddata["ivl"]
symbol_to_sp = {s: sp for s, sp in zip(symbols, subpathways)}
for tick, label in zip(ax.get_xticklabels(), order):
    sp = symbol_to_sp.get(label.upper(), "DNA_repair_other")
    tick.set_color(COLOR_MAP[sp])

ax.set_title(f"Hierarchical Clustering Dendrogram — colored by sub-pathway (Cophenetic: {coph:.4f})", fontsize=12)
ax.set_xlabel("Gene Symbol")
ax.set_ylabel("Distance")

legend_patches = [mpatches.Patch(color=c, label=n) for n, c in COLOR_MAP.items()]
ax.legend(handles=legend_patches, loc="upper right", fontsize=9)

plt.tight_layout()
out = "dendrogram_subpathway.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved to {out}")
