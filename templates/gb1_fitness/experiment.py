import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# GB1 combinatorial fitness landscape (Wu et al. 2016)
# Four positions (V39, D40, G41, V54) with all amino acid substitutions
# Data from the FLIP benchmark (Dallago et al. 2021), hosted on GitHub

GB1_ZIP_URL = "https://raw.githubusercontent.com/J-SNACKKB/FLIP/main/splits/gb1/four_mutations_full_data.csv.zip"

# GB1 wild-type sequence (IgG-binding domain, UniProt P06654)
GB1_WT = (
    "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
)
# Mutated positions in the WT (0-indexed): V39, D40, G41, V54
GB1_MUT_POSITIONS = [39, 40, 41, 54]


def download_gb1(data_dir):
    import urllib.request, zipfile, io
    os.makedirs(data_dir, exist_ok=True)
    dest = os.path.join(data_dir, "four_mutations_full_data.csv")
    if not os.path.exists(dest):
        response = urllib.request.urlopen(GB1_ZIP_URL)
        zf = zipfile.ZipFile(io.BytesIO(response.read()))
        zf.extractall(data_dir)
    return dest


def load_gb1(data_dir):
    import csv
    path = download_gb1(data_dir)
    sequences, fitnesses = [], []
    wt = list(GB1_WT)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row["Variants"]  # e.g. "ACGT" — four mutated AAs
            if len(variant) != 4:
                continue
            fitness = float(row["Fitness"])
            # Build full-length sequence with mutations applied
            seq = wt[:]
            for pos, aa in zip(GB1_MUT_POSITIONS, variant):
                seq[pos] = aa
            sequences.append("".join(seq))
            fitnesses.append(fitness)
    return sequences, np.array(fitnesses, dtype=np.float32)


# One-hot encode sequences (4 positions x 20 amino acids = 80-dim)
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA)}


def one_hot_encode(sequences):
    n = len(sequences)
    L = len(sequences[0])
    X = np.zeros((n, L * len(AA)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if aa in AA_IDX:
                X[i, j * len(AA) + AA_IDX[aa]] = 1.0
    return X


def get_esm2_embeddings(sequences, model_name, device, batch_size=64):
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i: i + batch_size]
        data = [(f"seq{j}", s) for j, s in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            results = model(tokens, repr_layers=[model.num_layers])
        # Mean-pool over sequence positions (excluding BOS/EOS)
        reps = results["representations"][model.num_layers]
        for k, seq in enumerate(batch_seqs):
            emb = reps[k, 1: len(seq) + 1].mean(0)
            all_embeddings.append(emb.cpu().float().numpy())
    return np.stack(all_embeddings)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = nn.functional.mse_loss(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds.append(model(X_batch).cpu().numpy())
            targets.append(y_batch.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mse = float(np.mean((preds - targets) ** 2))
    # Spearman correlation
    from scipy.stats import spearmanr
    spearman, _ = spearmanr(preds, targets)
    return {"mse": mse, "spearman": float(spearman)}


def run(out_dir, seed, train_frac=0.8, esm_model="esm2_t6_8M_UR50D",
        hidden_dims=(256, 128), dropout=0.1, lr=1e-3, epochs=50, batch_size=256):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(out_dir, "data")
    sequences, fitnesses = load_gb1(data_dir)

    # Log-transform fitness (standard for GB1)
    fitnesses = np.log1p(fitnesses)

    # Compute ESM-2 embeddings
    embeddings = get_esm2_embeddings(sequences, esm_model, device)
    input_dim = embeddings.shape[1]

    # Train/test split
    n = len(sequences)
    idx = np.random.permutation(n)
    n_train = int(n * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    X_train = torch.tensor(embeddings[train_idx])
    y_train = torch.tensor(fitnesses[train_idx])
    X_test = torch.tensor(embeddings[test_idx])
    y_test = torch.tensor(fitnesses[test_idx])

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

    final_info = {
        "final_train_loss": train_log[-1]["train_loss"],
        "final_val_mse": val_log[-1]["mse"],
        "final_val_spearman": val_log[-1]["spearman"],
        "best_val_spearman": float(max(v["spearman"] for v in val_log)),
        "n_train": int(n_train),
        "n_test": int(n - n_train),
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
    seeds = [0, 1, 2]
    all_results = {}
    final_infos_list = []

    for seed in seeds:
        print(f"Running seed {seed}")
        final_info, train_log, val_log = run(args.out_dir, seed)
        all_results[f"seed{seed}_final_info"] = final_info
        all_results[f"seed{seed}_train_log"] = train_log
        all_results[f"seed{seed}_val_log"] = val_log
        final_infos_list.append(final_info)

    keys = final_infos_list[0].keys()
    numeric_keys = [k for k in keys if isinstance(final_infos_list[0][k], (int, float))]
    final_infos = {
        "means": {k: float(np.mean([d[k] for d in final_infos_list])) for k in numeric_keys},
        "stderrs": {k: float(np.std([d[k] for d in final_infos_list]) / len(seeds)) for k in numeric_keys},
        "final_info_list": final_infos_list,
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)
