from pathlib import Path
import json

import numpy as np
import torch
from datasets import load_dataset


DATASET_ID = "AlexeyK421/ECG_Arrhythmia_Classification_Dataset"
OUTPUT_PATH = Path(__file__).resolve().parent / "ecg_arrhythmia_processed.pt"
META_PATH = Path(__file__).resolve().parent / "ecg_arrhythmia_metadata.json"
RANDOM_SEED = 20260525


def stratified_split(indices, labels, test_fraction, seed):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for cls in np.unique(labels):
        cls_idx = indices[labels == cls].copy()
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_fraction)))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def build_ecg_arrhythmia_dataset():
    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()

    feature_cols = [c for c in df.columns if c not in ("record", "type")]
    lead0_cols = [c for c in feature_cols if c.startswith("0_")]
    lead1_cols = [c for c in feature_cols if c.startswith("1_")]

    x = df[feature_cols].to_numpy(dtype=np.float32)
    labels_raw = df["type"].astype(str).to_numpy()
    class_names = sorted(np.unique(labels_raw).tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.array([class_to_idx[label] for label in labels_raw], dtype=np.int64)

    indices = np.arange(len(df), dtype=np.int64)
    trainval_idx, test_idx = stratified_split(indices, y, test_fraction=0.15, seed=RANDOM_SEED)
    train_idx, val_idx = stratified_split(trainval_idx, y[trainval_idx], test_fraction=0.1764705882, seed=RANDOM_SEED + 1)

    x_train = x[train_idx]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    x = (x - mean) / std

    payload = {
        "x_train": torch.from_numpy(x[train_idx]),
        "y_train": torch.from_numpy(y[train_idx]),
        "x_val": torch.from_numpy(x[val_idx]),
        "y_val": torch.from_numpy(y[val_idx]),
        "x_test": torch.from_numpy(x[test_idx]),
        "y_test": torch.from_numpy(y[test_idx]),
        "feature_cols": feature_cols,
        "lead0_cols": lead0_cols,
        "lead1_cols": lead1_cols,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "mean": torch.from_numpy(mean.astype(np.float32)).squeeze(0),
        "std": torch.from_numpy(std.astype(np.float32)).squeeze(0),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUTPUT_PATH)

    META_PATH.write_text(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "num_examples": int(len(df)),
                "num_features": len(feature_cols),
                "num_classes": len(class_names),
                "feature_cols": feature_cols,
                "lead0_cols": lead0_cols,
                "lead1_cols": lead1_cols,
                "class_names": class_names,
            },
            indent=2,
        )
    )

    print(f"Saved processed dataset to {OUTPUT_PATH}")


def main():
    if OUTPUT_PATH.exists():
        print(f"Dataset already exists at {OUTPUT_PATH}")
        return
    build_ecg_arrhythmia_dataset()


if __name__ == "__main__":
    main()

