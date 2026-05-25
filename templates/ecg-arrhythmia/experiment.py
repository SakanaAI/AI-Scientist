import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


DATA_PATH = Path("data") / "ecg_arrhythmia_processed.pt"
BATCH_SIZE = 1024
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
DROPOUT = 0.25
PATIENCE = 6
MODEL_VARIANT = os.environ.get("ECG_MODEL_VARIANT", "baseline")


def ensure_dataset_available():
    if DATA_PATH.exists():
        return
    from data.prepare import build_ecg_arrhythmia_dataset

    build_ecg_arrhythmia_dataset()


class TensorECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.float()
        self.y = y.long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class FlatMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x), None


class LeadAwareGatedFusionClassifier(nn.Module):
    def __init__(self, lead0_idx, lead1_idx, num_classes, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.register_buffer("lead0_idx", lead0_idx)
        self.register_buffer("lead1_idx", lead1_idx)
        branch_dim = hidden_dim // 2
        fused_dim = hidden_dim // 2
        self.lead0_branch = nn.Sequential(
            nn.Linear(len(lead0_idx), branch_dim),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, branch_dim),
            nn.GELU(),
        )
        self.lead1_branch = nn.Sequential(
            nn.Linear(len(lead1_idx), branch_dim),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, branch_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(branch_dim * 2, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, 1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(branch_dim, num_classes),
        )

    def forward(self, x):
        lead0 = x.index_select(dim=1, index=self.lead0_idx)
        lead1 = x.index_select(dim=1, index=self.lead1_idx)
        h0 = self.lead0_branch(lead0)
        h1 = self.lead1_branch(lead1)
        gate = self.gate(torch.cat([h0, h1], dim=-1))
        fused = gate * h0 + (1.0 - gate) * h1
        return self.head(fused), gate


class Trainer:
    def __init__(self, model, device, train_counts):
        self.model = model
        self.device = device
        self.train_info = []
        self.val_info = []
        self.start_time = time.time()
        self.best_val_loss = float("inf")
        self.best_state = None
        weights = torch.tensor(1.0 / np.maximum(train_counts, 1), dtype=torch.float32)
        weights = weights / weights.mean()
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

    def _step(self, batch, train=True):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        if train:
            self.optimizer.zero_grad()
        logits, _ = self.model(x)
        loss = self.criterion(logits, y)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return loss.detach(), logits.detach(), y.detach()

    def _evaluate_loader(self, loader):
        self.model.eval()
        losses = []
        logits_all = []
        targets_all = []
        gate_all = []
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits, gate = self.model(x)
                loss = self.criterion(logits, y)
                losses.append(loss.item())
                logits_all.append(logits.cpu())
                targets_all.append(y.cpu())
                if gate is not None:
                    gate_all.append(gate.cpu())

        logits_all = torch.cat(logits_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        probs = torch.softmax(logits_all, dim=1).numpy()
        preds = probs.argmax(axis=1)
        y_true = targets_all.numpy()
        num_classes = probs.shape[1]
        y_onehot = np.eye(num_classes, dtype=np.float32)[y_true]

        metrics = {
            "loss": float(np.mean(losses)),
            "macro_f1": float(f1_score(y_true, preds, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
            "macro_roc_auc": float(roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr")),
            "macro_avg_precision": float(average_precision_score(y_onehot, probs, average="macro")),
        }
        if gate_all:
            gate_tensor = torch.cat(gate_all, dim=0).squeeze(-1).numpy()
            metrics["mean_gate"] = float(np.mean(gate_tensor))
            metrics["std_gate"] = float(np.std(gate_tensor))
            per_class_gate = []
            for cls in range(num_classes):
                mask = y_true == cls
                per_class_gate.append(float(np.mean(gate_tensor[mask])) if mask.any() else float("nan"))
            metrics["per_class_gate"] = per_class_gate
        return metrics

    def fit(self, train_loader, val_loader):
        patience = 0
        for epoch in range(EPOCHS):
            self.model.train()
            train_losses = []
            train_logits = []
            train_targets = []
            for batch_idx, batch in enumerate(train_loader):
                loss, logits, targets = self._step(batch, train=True)
                train_losses.append(loss.item())
                train_logits.append(logits.cpu())
                train_targets.append(targets.cpu())

            train_logits = torch.cat(train_logits, dim=0)
            train_targets = torch.cat(train_targets, dim=0)
            train_probs = torch.softmax(train_logits, dim=1).numpy()
            train_preds = train_probs.argmax(axis=1)
            train_y = train_targets.numpy()

            train_metrics = {
                "loss": float(np.mean(train_losses)),
                "macro_f1": float(f1_score(train_y, train_preds, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(train_y, train_preds)),
            }
            val_metrics = self._evaluate_loader(val_loader)
            self.scheduler.step(val_metrics["loss"])

            self.train_info.append(
                {
                    "iter": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/macro_f1": train_metrics["macro_f1"],
                    "train/balanced_accuracy": train_metrics["balanced_accuracy"],
                }
            )
            self.val_info.append(
                {
                    "iter": epoch,
                    "train/loss": train_metrics["loss"],
                    "val/loss": val_metrics["loss"],
                    "val/macro_f1": val_metrics["macro_f1"],
                    "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val/macro_roc_auc": val_metrics["macro_roc_auc"],
                    "val/macro_avg_precision": val_metrics["macro_avg_precision"],
                }
            )

            print(
                f"Epoch {epoch:02d}: train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_state = deepcopy(self.model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("Early stopping triggered.")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

    def evaluate(self, loader):
        return self._evaluate_loader(loader)


def main():
    parser = argparse.ArgumentParser(description="Run ECG arrhythmia experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    ensure_dataset_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    data = torch.load(DATA_PATH, map_location="cpu")
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]
    class_names = data["class_names"]
    feature_cols = data["feature_cols"]
    lead0_cols = data["lead0_cols"]
    lead1_cols = data["lead1_cols"]

    lead0_idx = torch.tensor([feature_cols.index(name) for name in lead0_cols], dtype=torch.long)
    lead1_idx = torch.tensor([feature_cols.index(name) for name in lead1_cols], dtype=torch.long)

    train_loader = DataLoader(
        TensorECGDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorECGDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        TensorECGDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    counts = np.bincount(y_train.numpy(), minlength=len(class_names)).astype(np.float32)
    if MODEL_VARIANT == "lead_aware":
        model = LeadAwareGatedFusionClassifier(
            lead0_idx=lead0_idx.to(device),
            lead1_idx=lead1_idx.to(device),
            num_classes=len(class_names),
        ).to(device)
    else:
        model = FlatMLPClassifier(input_dim=x_train.shape[1], num_classes=len(class_names)).to(device)
    trainer = Trainer(model, device=device, train_counts=counts)
    print(f"Model variant: {MODEL_VARIANT}")

    print("Starting training")
    trainer.fit(train_loader, val_loader)

    print("Evaluating best checkpoint")
    test_metrics = trainer.evaluate(test_loader)

    results_dict = {
        "final_train_loss": trainer.train_info[-1]["train/loss"],
        "best_val_loss": trainer.best_val_loss,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_roc_auc": test_metrics["macro_roc_auc"],
        "test_macro_avg_precision": test_metrics["macro_avg_precision"],
        "total_train_time": time.time() - trainer.start_time,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    if "mean_gate" in test_metrics:
        results_dict["mean_gate"] = test_metrics["mean_gate"]
        results_dict["std_gate"] = test_metrics["std_gate"]
        results_dict["per_class_gate"] = test_metrics["per_class_gate"]

    scalar_results = {k: v for k, v in results_dict.items() if not isinstance(v, list)}
    formatted_results = {
        "ecg_arrhythmia": {
            "means": {f"{k}_mean": v for k, v in scalar_results.items()},
            "stderrs": {
                f"{k}_stderr": 0.0 for k in scalar_results.keys()
            },
            "final_info_dict": {
                k: [v] for k, v in scalar_results.items()
            },
            "diagnostics": {
                "class_names": class_names,
                "per_class_gate": test_metrics.get("per_class_gate"),
            },
        }
    }

    all_results = {
        "ecg_arrhythmia_final_info": formatted_results,
        "ecg_arrhythmia_train_info": trainer.train_info,
        "ecg_arrhythmia_val_info": trainer.val_info,
        "ecg_arrhythmia_class_names": class_names,
        "ecg_arrhythmia_test_metrics": test_metrics,
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(formatted_results, f)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)


if __name__ == "__main__":
    main()
