import os
import json
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torcheeg.datasets import BCICIV2aDataset
from torcheeg import transforms

TRAIN_SUBJECTS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07']
TEST_SUBJECTS = ['A08', 'A09']

# https://torcheeg.readthedocs.io/en/latest/generated/torcheeg.datasets.BCICIV2aDataset.html
class SubjectFilteredBCICIV2aDataset(BCICIV2aDataset):
    def __init__(self, subjects: list, *args, **kwargs):
        self.subjects = subjects
        super().__init__(*args, **kwargs)

    def set_records(self, root_path: str, **kwargs):
        all_files = super().set_records(root_path)
        return [f for f in all_files if os.path.basename(f)[:3] in self.subjects]

# https://arxiv.org/pdf/1611.08024
class EEGNet(nn.Module):
    def __init__(self, num_classes=4, num_channels=22, num_samples=1750,
                 dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False), # depth-wise
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False), # point-wise: 1x1 conv to combine depthwise outputs
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * ((num_samples // 32)), num_classes)
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.second_conv(x)
        x = self.separable_conv(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()
    
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    # logging directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # dataset
    train_dataset = SubjectFilteredBCICIV2aDataset(
        subjects=TRAIN_SUBJECTS,
        root_path='./data',
        online_transform=transforms.Compose([
            transforms.To2d(),
            transforms.ToTensor()
        ]),
        label_transform=transforms.Compose([
            transforms.Select('label'),
            transforms.Lambda(lambda x: x - 1)
        ])
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # main Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGNet(num_classes=4, kernel_length=125).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_accuracies = []

    start_time = time.time()
    for epoch in tqdm(range(100)):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss)

        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        val_accuracies.append(acc)

    end_time = time.time()
    training_time = end_time - start_time

    # test the model
    test_dataset = SubjectFilteredBCICIV2aDataset(
        subjects=TEST_SUBJECTS,
        root_path='./data/BCICIV_2a_mat_evaluate',
        online_transform=transforms.Compose([
            transforms.To2d(),
            transforms.ToTensor()
        ]),
        label_transform=transforms.Compose([
            transforms.Select('label'),
            transforms.Lambda(lambda x: x - 1)
        ])
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    results = []
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating"):
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().tolist()
            labels = y.tolist()
            for p, l in zip(preds, labels):
                results.append({"pred": int(p), "label": int(l)})

    labels = [r["label"] for r in results]
    preds = [r["pred"] for r in results]
    test_acc = accuracy_score(labels, preds)
    conf_mat = confusion_matrix(labels, preds).tolist()

    # save the outputs (structure for a single run)
    final_info = {
        "eegnet": {
            "means": {
                "training_time": training_time,
                "final_val_accuracy": val_accuracies[-1],
                "test_accuracy": test_acc,
            },
            "stderrs": {
                "training_time_stderr": 0.0,
                "final_val_accuracy_stderr": 0.0,
                "test_accuracy_stderr": 0.0,
            },
            "final_info_dict": {
                "training_time": [training_time],
                "final_val_accuracy": [val_accuracies[-1]],
                "test_accuracy": [test_acc],
            }
        }
    }
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f, indent=2)

    all_results = {
        "eegnet_train_losses": train_losses,
        "eegnet_val_accuracies": val_accuracies,
        "eegnet_test_results": results,
        "eegnet_confusion_matrix": conf_mat
    }
    with open(os.path.join(args.out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))