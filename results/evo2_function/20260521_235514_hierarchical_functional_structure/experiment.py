import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# KEGG gene list retrieval
# ---------------------------------------------------------------------------

# DNA repair pathways: MMR, NER, HR, BER
DNA_REPAIR_PATHWAYS = ["hsa03430", "hsa03420", "hsa03440", "hsa03410"]

# Tumor suppressor genes: curated from CGC (Cosmic Gene Census) Tier 1 TSGs
# supplemented with canonical literature (RB1, TP53, APC, PTEN, VHL, etc.)
# We use KEGG pathway hsa05200 (Pathways in Cancer) and filter to known TSGs
# via a curated seed list cross-referenced with KEGG.
TSG_SEED_SYMBOLS = [
    "TP53", "RB1", "APC", "PTEN", "VHL", "BRCA1", "BRCA2", "NF1", "NF2",
    "WT1", "CDKN2A", "CDKN1B", "CDKN1C", "MLH1", "MSH2", "MSH6", "PMS2",
    "STK11", "TSC1", "TSC2", "SMAD4", "SMAD2", "RET", "MEN1", "PTCH1",
    "SUFU", "BAP1", "PALB2", "ATM", "CHEK2", "CDH1", "RUNX1", "TET2",
    "DNMT3A", "ASXL1", "EZH2", "KDM6A", "ARID1A", "ARID1B", "PBRM1",
    "SETD2", "KDM5C", "FBXW7", "PPP2R1A", "PIK3R1", "INPP4B",
]


def fetch_kegg_genes(pathway_id, retries=3, delay=1.0):
    import urllib.request
    url = f"https://rest.kegg.jp/link/hsa/{pathway_id}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode()
            genes = []
            for line in text.strip().split("\n"):
                if not line:
                    continue
                kegg_id = line.split("\t")[1].strip()  # e.g. hsa:5111
                genes.append(kegg_id)
            return genes
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e


def kegg_id_to_symbol(kegg_id, retries=3, delay=0.5):
    """Convert hsa:NNNNN to gene symbol via KEGG list endpoint."""
    import urllib.request
    url = f"https://rest.kegg.jp/list/{kegg_id}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                text = resp.read().decode().strip()
            if not text:
                return None
            # Format: "hsa:5111\tPCNA, ATLD2; proliferating cell nuclear antigen"
            parts = text.split("\t")
            if len(parts) < 2:
                return None
            symbol = parts[1].split(",")[0].split(";")[0].strip()
            return symbol
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def get_dna_repair_symbols():
    seen_ids = set()
    symbols = []
    for pathway in DNA_REPAIR_PATHWAYS:
        try:
            kegg_ids = fetch_kegg_genes(pathway)
            for kid in kegg_ids:
                if kid in seen_ids:
                    continue
                seen_ids.add(kid)
                sym = kegg_id_to_symbol(kid)
                if sym:
                    symbols.append(sym.upper())
                time.sleep(0.2)
        except Exception as e:
            print(f"Warning: could not fetch {pathway}: {e}")
    return list(set(symbols))


# ---------------------------------------------------------------------------
# Ensembl sequence retrieval
# ---------------------------------------------------------------------------

ENSEMBL_REST = "https://rest.ensembl.org"
PROMOTER_BP = 2000


def symbol_to_ensembl(symbol, retries=3, delay=1.0):
    import urllib.request
    url = f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{symbol}?content-type=application/json"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            return data
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def fetch_promoter_sequence(chrom, tss, strand, bp=PROMOTER_BP, retries=3, delay=1.0):
    """Fetch bp upstream of TSS on the correct strand."""
    import urllib.request
    if strand == 1:
        start = max(1, tss - bp)
        end = tss
        strand_param = 1
    else:
        start = tss
        end = tss + bp
        strand_param = -1
    url = (
        f"{ENSEMBL_REST}/sequence/region/human/{chrom}:{start}..{end}:{strand_param}"
        f"?content-type=application/json"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            return data["seq"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def fetch_cds_sequence(ensembl_gene_id, retries=3, delay=1.0):
    """Fetch the canonical (longest) CDS for a gene."""
    import urllib.request
    url = (
        f"{ENSEMBL_REST}/sequence/id/{ensembl_gene_id}"
        f"?content-type=application/json&type=cds&multiple_sequences=1"
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            if isinstance(data, list):
                # Pick longest CDS
                return max(data, key=lambda x: len(x["seq"]))["seq"]
            return data["seq"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
    return None


def get_gene_sequence(symbol):
    """Return promoter + CDS concatenated for a gene symbol. Returns None on failure."""
    info = symbol_to_ensembl(symbol)
    if info is None:
        return None
    chrom = info["seq_region_name"]
    strand = info["strand"]
    # TSS = start for + strand, end for - strand
    tss = info["start"] if strand == 1 else info["end"]
    ensembl_id = info["id"]

    promoter = fetch_promoter_sequence(chrom, tss, strand)
    time.sleep(0.3)
    cds = fetch_cds_sequence(ensembl_id)
    time.sleep(0.3)

    if promoter is None or cds is None:
        return None
    return (promoter + cds).upper()


# ---------------------------------------------------------------------------
# Evo2 embedding extraction
# ---------------------------------------------------------------------------

EVO2_LAYER = "blocks.28.mlp.l3"  # late layer recommended by Arc Institute paper


def load_evo2(model_name="evo2_7b", device="cuda"):
    from evo2 import Evo2
    model = Evo2(model_name)
    return model


def get_evo2_embeddings(sequences, model, device, max_len=8192, batch_size=4):
    """Extract mean-pooled late-layer embeddings from Evo2."""
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i: i + batch_size]
        batch_embs = []
        for seq in batch:
            # Truncate to max_len (promoter+CDS can be very long)
            seq = seq[:max_len]
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq),
                dtype=torch.int,
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                _, emb_dict = model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[EVO2_LAYER],
                )
            # emb shape: (1, seq_len, hidden_dim) — mean pool over positions
            emb = emb_dict[EVO2_LAYER].squeeze(0).mean(0).float().cpu().numpy()
            batch_embs.append(emb)
        embeddings.extend(batch_embs)
    return np.stack(embeddings)


# ---------------------------------------------------------------------------
# MLP classifier
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))  # binary: 1 output + BCEWithLogitsLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            all_logits.append(model(X_batch).cpu().numpy())
            all_labels.append(y_batch.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(float)

    acc = float(np.mean(preds == labels))
    # AUROC
    from sklearn.metrics import roc_auc_score, f1_score
    auroc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5
    f1 = float(f1_score(labels, preds, zero_division=0))
    return {"accuracy": acc, "auroc": auroc, "f1": f1}


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def build_dataset(cache_dir, force_refetch=False):
    """Fetch gene sequences and labels, cache to disk."""
    cache_path = os.path.join(cache_dir, "dataset.json")
    if os.path.exists(cache_path) and not force_refetch:
        with open(cache_path) as f:
            return json.load(f)

    # Fall back to the baseline template cache if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_cache = os.path.join(script_dir, "run_0", "data", "dataset.json")
    if os.path.exists(baseline_cache) and not force_refetch:
        print(f"Using baseline cache from {baseline_cache}")
        os.makedirs(cache_dir, exist_ok=True)
        import shutil
        shutil.copy(baseline_cache, cache_path)
        with open(cache_path) as f:
            return json.load(f)

    os.makedirs(cache_dir, exist_ok=True)
    dataset = []

    # DNA repair genes (label=0)
    print("Fetching DNA repair gene list from KEGG...")
    repair_symbols = get_dna_repair_symbols()
    print(f"  Found {len(repair_symbols)} DNA repair genes")
    for sym in repair_symbols:
        print(f"  Fetching sequence for {sym}...")
        seq = get_gene_sequence(sym)
        if seq:
            dataset.append({"symbol": sym, "label": 0, "label_name": "DNA_repair", "seq": seq})

    # Tumor suppressor genes (label=1)
    print(f"Fetching sequences for {len(TSG_SEED_SYMBOLS)} tumor suppressor genes...")
    for sym in TSG_SEED_SYMBOLS:
        print(f"  Fetching sequence for {sym}...")
        seq = get_gene_sequence(sym)
        if seq:
            dataset.append({"symbol": sym, "label": 1, "label_name": "tumor_suppressor", "seq": seq})

    with open(cache_path, "w") as f:
        json.dump(dataset, f)

    n_repair = sum(1 for d in dataset if d["label"] == 0)
    n_tsg = sum(1 for d in dataset if d["label"] == 1)
    print(f"Dataset: {n_repair} DNA repair, {n_tsg} tumor suppressor genes")
    return dataset


def annotate_subpathways(dataset):
    """Annotate DNA repair genes with their sub-pathway labels."""
    pathway_map = {
        "hsa03430": "MMR",
        "hsa03420": "NER", 
        "hsa03440": "HR",
        "hsa03410": "BER"
    }
    
    # Build symbol to pathway mapping
    symbol_to_pathways = {}
    for pathway_id, pathway_name in pathway_map.items():
        try:
            kegg_ids = fetch_kegg_genes(pathway_id)
            for kid in kegg_ids:
                sym = kegg_id_to_symbol(kid)
                if sym:
                    sym = sym.upper()
                    if sym not in symbol_to_pathways:
                        symbol_to_pathways[sym] = []
                    symbol_to_pathways[sym].append(pathway_name)
                time.sleep(0.2)
        except Exception as e:
            print(f"Warning: could not fetch {pathway_id}: {e}")
    
    # Annotate dataset
    for item in dataset:
        if item["label"] == 0:  # DNA repair gene
            symbol = item["symbol"]
            item["subpathways"] = symbol_to_pathways.get(symbol, [])
        else:
            item["subpathways"] = []
    
    return dataset


def identify_bridge_genes(dataset):
    """Find genes that appear in both DNA repair and TSG lists."""
    repair_symbols = {d["symbol"] for d in dataset if d["label"] == 0}
    tsg_symbols = {d["symbol"] for d in dataset if d["label"] == 1}
    bridge_genes = repair_symbols & tsg_symbols
    
    for item in dataset:
        item["is_bridge"] = item["symbol"] in bridge_genes
    
    return dataset, list(bridge_genes)


def hierarchical_clustering_analysis(embeddings, labels, symbols, subpathways, out_dir):
    """Perform hierarchical clustering and compute cophenetic correlation."""
    from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
    from scipy.spatial.distance import pdist, squareform
    import matplotlib.pyplot as plt
    
    # Compute pairwise distances
    distances = pdist(embeddings, metric='cosine')
    
    # Perform Ward linkage
    Z = linkage(distances, method='ward')
    
    # Compute cophenetic correlation
    c, coph_dists = cophenet(Z, distances)
    print(f"Cophenetic correlation: {c:.4f}")
    
    # Create dendrogram
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Color by class
    label_colors = ['blue' if l == 0 else 'red' for l in labels]
    
    dend = dendrogram(Z, labels=symbols, ax=ax, leaf_font_size=6)
    ax.set_title(f'Hierarchical Clustering Dendrogram (Cophenetic corr: {c:.4f})')
    ax.set_xlabel('Gene Symbol')
    ax.set_ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dendrogram.png'), dpi=150)
    plt.close()
    
    return c, Z


def subpathway_coherence_analysis(embeddings, labels, subpathways, out_dir, n_permutations=1000):
    """Compute sub-pathway coherence with permutation test."""
    from scipy.spatial.distance import cosine
    from scipy.stats import ttest_ind
    
    # Filter to DNA repair genes only
    repair_mask = labels == 0
    repair_embeddings = embeddings[repair_mask]
    repair_subpathways = [sp for sp, is_repair in zip(subpathways, repair_mask) if is_repair]
    
    # Get primary sub-pathway for each gene (take first if multiple)
    primary_pathways = [sp[0] if sp else None for sp in repair_subpathways]
    
    # Count genes per pathway
    pathway_counts = {}
    for pw in primary_pathways:
        if pw:
            pathway_counts[pw] = pathway_counts.get(pw, 0) + 1
    
    print(f"Sub-pathway counts: {pathway_counts}")
    
    # Compute within vs between pathway distances
    results = {}
    for pathway_name in ["MMR", "NER", "HR", "BER"]:
        pathway_indices = [i for i, pw in enumerate(primary_pathways) if pw == pathway_name]
        
        if len(pathway_indices) < 3:
            print(f"Skipping {pathway_name}: only {len(pathway_indices)} genes")
            continue
        
        # Within-pathway distances
        within_dists = []
        for i in range(len(pathway_indices)):
            for j in range(i+1, len(pathway_indices)):
                idx_i, idx_j = pathway_indices[i], pathway_indices[j]
                dist = cosine(repair_embeddings[idx_i], repair_embeddings[idx_j])
                within_dists.append(dist)
        
        # Between-pathway distances (to other DNA repair genes)
        between_dists = []
        other_indices = [i for i, pw in enumerate(primary_pathways) if pw and pw != pathway_name]
        for idx_i in pathway_indices:
            for idx_j in other_indices:
                dist = cosine(repair_embeddings[idx_i], repair_embeddings[idx_j])
                between_dists.append(dist)
        
        mean_within = np.mean(within_dists)
        mean_between = np.mean(between_dists)
        
        # Permutation test
        observed_diff = mean_between - mean_within
        null_diffs = []
        
        for _ in range(n_permutations):
            # Shuffle pathway labels
            shuffled_pathways = primary_pathways.copy()
            np.random.shuffle(shuffled_pathways)
            
            shuffled_pathway_indices = [i for i, pw in enumerate(shuffled_pathways) if pw == pathway_name]
            if len(shuffled_pathway_indices) < 3:
                continue
            
            # Within-pathway distances (shuffled)
            shuffled_within = []
            for i in range(len(shuffled_pathway_indices)):
                for j in range(i+1, len(shuffled_pathway_indices)):
                    idx_i, idx_j = shuffled_pathway_indices[i], shuffled_pathway_indices[j]
                    dist = cosine(repair_embeddings[idx_i], repair_embeddings[idx_j])
                    shuffled_within.append(dist)
            
            # Between-pathway distances (shuffled)
            shuffled_other_indices = [i for i, pw in enumerate(shuffled_pathways) if pw and pw != pathway_name]
            shuffled_between = []
            for idx_i in shuffled_pathway_indices:
                for idx_j in shuffled_other_indices:
                    dist = cosine(repair_embeddings[idx_i], repair_embeddings[idx_j])
                    shuffled_between.append(dist)
            
            if shuffled_within and shuffled_between:
                null_diff = np.mean(shuffled_between) - np.mean(shuffled_within)
                null_diffs.append(null_diff)
        
        p_value = np.mean([d >= observed_diff for d in null_diffs]) if null_diffs else 1.0
        
        results[pathway_name] = {
            "n_genes": len(pathway_indices),
            "mean_within_dist": float(mean_within),
            "mean_between_dist": float(mean_between),
            "coherence_score": float(observed_diff),
            "p_value": float(p_value)
        }
        
        print(f"{pathway_name}: within={mean_within:.4f}, between={mean_between:.4f}, diff={observed_diff:.4f}, p={p_value:.4f}")
    
    return results


def bridge_gene_analysis(embeddings, labels, symbols, is_bridge, out_dir):
    """Analyze bridge gene positioning relative to class centroids."""
    from scipy.spatial.distance import cosine
    from scipy.stats import ttest_ind
    
    # Compute class centroids
    repair_mask = labels == 0
    tsg_mask = labels == 1
    
    repair_centroid = embeddings[repair_mask].mean(axis=0)
    tsg_centroid = embeddings[tsg_mask].mean(axis=0)
    
    # Find bridge genes and other TSGs
    bridge_mask = np.array(is_bridge) & tsg_mask
    other_tsg_mask = (~np.array(is_bridge)) & tsg_mask
    
    bridge_symbols = [s for s, b in zip(symbols, bridge_mask) if b]
    print(f"Bridge genes: {bridge_symbols}")
    
    # Compute distances to centroids for bridge genes
    bridge_to_repair = []
    bridge_to_tsg = []
    for emb in embeddings[bridge_mask]:
        bridge_to_repair.append(cosine(emb, repair_centroid))
        bridge_to_tsg.append(cosine(emb, tsg_centroid))
    
    # Compute distances to centroids for other TSGs
    other_tsg_to_repair = []
    other_tsg_to_tsg = []
    for emb in embeddings[other_tsg_mask]:
        other_tsg_to_repair.append(cosine(emb, repair_centroid))
        other_tsg_to_tsg.append(cosine(emb, tsg_centroid))
    
    # Statistical test
    if bridge_to_repair and other_tsg_to_repair:
        t_stat, p_value = ttest_ind(bridge_to_repair, other_tsg_to_repair)
        
        results = {
            "n_bridge_genes": len(bridge_symbols),
            "bridge_genes": bridge_symbols,
            "bridge_to_repair_mean": float(np.mean(bridge_to_repair)),
            "bridge_to_tsg_mean": float(np.mean(bridge_to_tsg)),
            "other_tsg_to_repair_mean": float(np.mean(other_tsg_to_repair)),
            "other_tsg_to_tsg_mean": float(np.mean(other_tsg_to_tsg)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value)
        }
        
        print(f"Bridge genes closer to DNA repair? t={t_stat:.4f}, p={p_value:.4f}")
        print(f"Bridge→Repair: {results['bridge_to_repair_mean']:.4f}, Other TSG→Repair: {results['other_tsg_to_repair_mean']:.4f}")
    else:
        results = {"error": "Insufficient bridge genes or TSGs"}
    
    return results


def umap_visualization(embeddings, labels, symbols, subpathways, is_bridge, out_dir):
    """Generate UMAP visualization colored by sub-pathway."""
    try:
        from umap import UMAP
    except ImportError:
        print("UMAP not installed, skipping visualization")
        return
    
    import matplotlib.pyplot as plt
    
    # Compute UMAP
    reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Color by class
    ax = axes[0]
    repair_mask = labels == 0
    tsg_mask = labels == 1
    
    ax.scatter(embedding_2d[repair_mask, 0], embedding_2d[repair_mask, 1], 
               c='blue', label='DNA Repair', alpha=0.6, s=50)
    ax.scatter(embedding_2d[tsg_mask, 0], embedding_2d[tsg_mask, 1], 
               c='red', label='TSG', alpha=0.6, s=50)
    
    # Highlight bridge genes
    bridge_mask = np.array(is_bridge)
    if bridge_mask.any():
        ax.scatter(embedding_2d[bridge_mask, 0], embedding_2d[bridge_mask, 1], 
                   c='gold', edgecolors='black', label='Bridge Genes', alpha=1.0, s=100, marker='*')
    
    ax.set_title('UMAP: Gene Classes')
    ax.legend()
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # Plot 2: Color by sub-pathway (DNA repair only)
    ax = axes[1]
    pathway_colors = {'MMR': 'green', 'NER': 'orange', 'HR': 'purple', 'BER': 'cyan', None: 'gray'}
    
    for i, (emb_2d, label, sp_list) in enumerate(zip(embedding_2d, labels, subpathways)):
        if label == 0:  # DNA repair
            sp = sp_list[0] if sp_list else None
            color = pathway_colors.get(sp, 'gray')
            ax.scatter(emb_2d[0], emb_2d[1], c=color, alpha=0.6, s=50)
        else:  # TSG
            ax.scatter(emb_2d[0], emb_2d[1], c='lightgray', alpha=0.3, s=30)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=pathway_colors[pw], label=pw) 
                      for pw in ['MMR', 'NER', 'HR', 'BER']]
    legend_elements.append(Patch(facecolor='lightgray', label='TSG'))
    ax.legend(handles=legend_elements)
    ax.set_title('UMAP: DNA Repair Sub-Pathways')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'umap_analysis.png'), dpi=150)
    plt.close()


def run(out_dir, seed, model_name="evo2_7b", hidden_dims=(256, 128),
        dropout=0.2, lr=1e-3, epochs=10, batch_size=16, train_frac=0.8):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cached dataset and embeddings from run_0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_data_path = os.path.join(script_dir, "run_0", "data", "dataset.json")
    baseline_emb_path = os.path.join(script_dir, "run_0", "data", f"embeddings_{model_name}.npy")
    
    if not os.path.exists(baseline_data_path) or not os.path.exists(baseline_emb_path):
        print("ERROR: Baseline data not found. Please run baseline first.")
        return {}, [], []
    
    # Load dataset
    with open(baseline_data_path) as f:
        dataset = json.load(f)
    
    # Load embeddings
    embeddings = np.load(baseline_emb_path)
    
    print(f"Loaded {len(dataset)} genes with embeddings of shape {embeddings.shape}")
    
    # Step 1: Annotate sub-pathways
    print("\n=== Step 1: Annotating sub-pathways ===")
    dataset = annotate_subpathways(dataset)
    
    # Step 2: Identify bridge genes
    print("\n=== Step 2: Identifying bridge genes ===")
    dataset, bridge_genes = identify_bridge_genes(dataset)
    print(f"Found {len(bridge_genes)} bridge genes: {bridge_genes}")
    
    # Extract arrays for analysis
    labels = np.array([d["label"] for d in dataset], dtype=np.float32)
    symbols = [d["symbol"] for d in dataset]
    subpathways = [d["subpathways"] for d in dataset]
    is_bridge = [d["is_bridge"] for d in dataset]
    
    # Step 3: Hierarchical clustering
    print("\n=== Step 3: Hierarchical clustering ===")
    cophenetic_corr, linkage_matrix = hierarchical_clustering_analysis(
        embeddings, labels, symbols, subpathways, out_dir
    )
    
    # Step 4: Sub-pathway coherence
    print("\n=== Step 4: Sub-pathway coherence analysis ===")
    coherence_results = subpathway_coherence_analysis(
        embeddings, labels, subpathways, out_dir, n_permutations=1000
    )
    
    # Step 5: Bridge gene analysis
    print("\n=== Step 5: Bridge gene analysis ===")
    bridge_results = bridge_gene_analysis(
        embeddings, labels, symbols, is_bridge, out_dir
    )
    
    # Step 6: UMAP visualization
    print("\n=== Step 6: UMAP visualization ===")
    umap_visualization(embeddings, labels, symbols, subpathways, is_bridge, out_dir)
    
    # Compile final results
    final_info = {
        "cophenetic_correlation": float(cophenetic_corr),
        "n_bridge_genes": len(bridge_genes),
        "bridge_genes": bridge_genes,
        "subpathway_coherence": coherence_results,
        "bridge_gene_analysis": bridge_results,
        "n_total_genes": len(dataset),
        "n_dna_repair": int((labels == 0).sum()),
        "n_tsg": int((labels == 1).sum()),
    }
    
    print("\n=== Final Results ===")
    print(json.dumps(final_info, indent=2))
    
    with open(os.path.join(out_dir, f"final_info_seed{seed}.json"), "w") as f:
        json.dump(final_info, f, indent=2)
    
    # Return empty logs since we're not training
    return final_info, [], []


parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str, default="run_0")
args = parser.parse_args()

if __name__ == "__main__":
    seeds = [0]
    all_results = {}
    final_infos_list = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        final_info, train_log, val_log = run(args.out_dir, seed)
        all_results[f"seed{seed}_final_info"] = final_info
        all_results[f"seed{seed}_train_log"] = train_log
        all_results[f"seed{seed}_val_log"] = val_log
        final_infos_list.append(final_info)

    # Handle both training runs and analysis runs
    numeric_keys = [k for k in final_infos_list[0] if isinstance(final_infos_list[0][k], (int, float))]
    if numeric_keys:
        final_infos = {
            "means": {k: float(np.mean([d[k] for d in final_infos_list])) for k in numeric_keys},
            "stderrs": {k: float(np.std([d[k] for d in final_infos_list]) / len(seeds)) for k in numeric_keys},
            "final_info_list": final_infos_list,
        }
    else:
        final_infos = {"final_info_list": final_infos_list}

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f, indent=2)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)
