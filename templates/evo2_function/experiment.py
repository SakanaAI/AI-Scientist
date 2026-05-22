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


def run(out_dir, seed, model_name="evo2_7b", hidden_dims=(256, 128),
        dropout=0.2, lr=1e-3, epochs=10, batch_size=16, train_frac=0.8):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build / load dataset
    cache_dir = os.path.join(out_dir, "data")
    dataset = build_dataset(cache_dir)

    sequences = [d["seq"] for d in dataset]
    labels = np.array([d["label"] for d in dataset], dtype=np.float32)
    symbols = [d["symbol"] for d in dataset]

    print(f"Total genes: {len(sequences)} | DNA repair: {int((labels==0).sum())} | TSG: {int((labels==1).sum())}")

    # Extract Evo2 embeddings
    emb_cache = os.path.join(out_dir, "data", f"embeddings_{model_name}.npy")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_emb_cache = os.path.join(script_dir, "run_0", "data", f"embeddings_{model_name}.npy")
    if os.path.exists(emb_cache):
        print("Loading cached embeddings...")
        embeddings = np.load(emb_cache)
    elif os.path.exists(baseline_emb_cache):
        print(f"Using baseline embeddings cache from {baseline_emb_cache}")
        import shutil
        shutil.copy(baseline_emb_cache, emb_cache)
        embeddings = np.load(emb_cache)
    else:
        print(f"Extracting Evo2 embeddings ({model_name})...")
        evo2 = load_evo2(model_name, str(device))
        embeddings = get_evo2_embeddings(sequences, evo2, device)
        np.save(emb_cache, embeddings)
        del evo2
        torch.cuda.empty_cache()

    input_dim = embeddings.shape[1]
    print(f"Embedding shape: {embeddings.shape}")

    # Train/test split
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(sequences))
    n_train = int(len(sequences) * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train = torch.tensor(embeddings[train_idx])
    y_train = torch.tensor(labels[train_idx])
    X_test = torch.tensor(embeddings[test_idx])
    y_test = torch.tensor(labels[test_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    model = MLP(input_dim, list(hidden_dims), dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_log, val_log = [], []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, test_loader, device)
        scheduler.step()
        train_log.append({"epoch": epoch, "train_loss": train_loss})
        val_log.append({"epoch": epoch, **val_metrics})
        if epoch % 5 == 0:
            print(f"  epoch {epoch}: loss={train_loss:.4f} acc={val_metrics['accuracy']:.3f} auroc={val_metrics['auroc']:.3f} f1={val_metrics['f1']:.3f}")

    final_info = {
        "final_train_loss": train_log[-1]["train_loss"],
        "final_val_accuracy": val_log[-1]["accuracy"],
        "final_val_auroc": val_log[-1]["auroc"],
        "final_val_f1": val_log[-1]["f1"],
        "best_val_auroc": float(max(v["auroc"] for v in val_log)),
        "best_val_accuracy": float(max(v["accuracy"] for v in val_log)),
        "n_train": int(n_train),
        "n_test": int(len(sequences) - n_train),
        "n_dna_repair": int((labels == 0).sum()),
        "n_tsg": int((labels == 1).sum()),
        "input_dim": int(input_dim),
    }
    print(final_info)

    with open(os.path.join(out_dir, f"final_info_seed{seed}.json"), "w") as f:
        json.dump(final_info, f)

    return final_info, train_log, val_log


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

    numeric_keys = [k for k in final_infos_list[0] if isinstance(final_infos_list[0][k], (int, float))]
    final_infos = {
        "means": {k: float(np.mean([d[k] for d in final_infos_list])) for k in numeric_keys},
        "stderrs": {k: float(np.std([d[k] for d in final_infos_list]) / len(seeds)) for k in numeric_keys},
        "final_info_list": final_infos_list,
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)
