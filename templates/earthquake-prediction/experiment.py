import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Model hyperparameters
EMB_SIZE = 16
HID_SIZE = 32
N_CELLS_HOR = 200
N_CELLS_VER = 250
OBSERVED_DAYS = 64
DAYS_TO_PREDICT_AFTER = 10
DAYS_TO_PREDICT_BEFORE = 50
TESTING_DAYS = 1000
HEAVY_QUAKE_THRES = 3.5

# Training hyperparameters
BATCH_SIZE = 1
N_CYCLES = 10
QUEUE_LENGTH = 50
LEARNING_RATE = 0.0003
LR_DECAY = 10.
EARTHQUAKE_WEIGHT = 10000.


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.CONV = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.BNORM = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=False)
        self.MAXPOOL = nn.MaxPool2d(3, stride=1, padding=1, dilation=1)

    def forward(self, x):
        x = self.CONV(x)
        x = self.BNORM(x)
        x = self.MAXPOOL(x)
        return x


class LSTMCell(nn.Module):
    def __init__(self, frequency_map, embedding_size=EMB_SIZE, hidden_state_size=HID_SIZE,
                 n_cells_hor=N_CELLS_HOR, n_cells_ver=N_CELLS_VER, device=torch.device('cpu')):
        super().__init__()

        self.n_cells_hor = n_cells_hor
        self.n_cells_ver = n_cells_ver

        # Move frequency map to device immediately
        self.freq_map = torch.cat([1 - frequency_map, frequency_map], dim=0).unsqueeze(0).to(device)
        self.freq_map.requires_grad = True

        self.emb_size = embedding_size
        self.hid_size = hidden_state_size

        self.embedding = nn.Sequential(
            ConvBlock(1, self.emb_size, 3),
            nn.ReLU(),
            ConvBlock(self.emb_size, self.emb_size, 3)
        )

        self.hidden_to_result = nn.Sequential(
            ConvBlock(hidden_state_size, 2, kernel_size=3),
            nn.Softmax(dim=1)
        )

        self.f_t = nn.Sequential(
            ConvBlock(self.hid_size + self.emb_size, self.hid_size, 3),
            nn.Sigmoid()
        )

        self.i_t = nn.Sequential(
            ConvBlock(self.hid_size + self.emb_size, self.hid_size, 3),
            nn.Sigmoid()
        )

        self.c_t = nn.Sequential(
            ConvBlock(self.hid_size + self.emb_size, self.hid_size, 3),
            nn.Tanh()
        )

        self.o_t = nn.Sequential(
            ConvBlock(self.hid_size + self.emb_size, self.hid_size, 3),
            nn.Sigmoid()
        )

        # Move all components to device
        self.to(device)

    def forward(self, x, prev_state):
        prev_c, prev_h = prev_state
        x_emb = self.embedding(x)

        x_and_h = torch.cat([prev_h, x_emb], dim=1)

        f_i = self.f_t(x_and_h)
        i_i = self.i_t(x_and_h)
        c_i = self.c_t(x_and_h)
        o_i = self.o_t(x_and_h)

        next_c = prev_c * f_i + i_i * c_i
        next_h = torch.tanh(next_c) * o_i

        correction = self.hidden_to_result(next_h)[:, 0, :, :]
        prediction = torch.cat([self.freq_map for _ in range(correction.shape[0])], dim=0)

        prediction_new = prediction.clone()
        prediction_new[:, 0, :, :] = prediction[:, 0, :, :] - correction
        prediction_new[:, 1, :, :] = prediction[:, 1, :, :] + correction

        return (next_c, next_h), prediction_new

    def init_state(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.hid_size, self.n_cells_hor, self.n_cells_ver, device=device),
            torch.zeros(batch_size, self.hid_size, self.n_cells_hor, self.n_cells_ver, device=device)
        )


class Dataset_RNN_Train(Dataset):
    def __init__(self, celled_data, device='cpu'):
        self.data = celled_data[0:(celled_data.shape[0] - TESTING_DAYS)].cpu()
        self.size = (self.data.shape[0] - DAYS_TO_PREDICT_BEFORE)
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_data = self.data[idx].clone()
        target_data = self.data[
                      (idx + DAYS_TO_PREDICT_AFTER):(idx + DAYS_TO_PREDICT_BEFORE)
                      ]

        # Calculate target and reshape to 2D
        target = torch.sum(
            target_data > HEAVY_QUAKE_THRES,
            dim=0,
            keepdim=True
        ).squeeze(0) > 0

        return input_data, target.long()  # Ensure target is long type


class Dataset_RNN_Test(Dataset):
    """Dataset class for testing earthquake prediction model."""

    def __init__(self, celled_data):
        self.data = celled_data[(celled_data.shape[0] - TESTING_DAYS):(celled_data.shape[0])]
        self.size = (self.data.shape[0] - DAYS_TO_PREDICT_BEFORE)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            self.data[idx],
            torch.sum(
                self.data[
                (idx + DAYS_TO_PREDICT_AFTER):(idx + DAYS_TO_PREDICT_BEFORE)
                ] > HEAVY_QUAKE_THRES,
                dim=0,
                keepdim=True
            ).squeeze(0) > 0
        )


class Trainer:
    def __init__(self, model, device, learning_rate=LEARNING_RATE,
                 earthquake_weight=EARTHQUAKE_WEIGHT):
        self.model = model
        self.global_iter = 0  # Track global iterations across phases
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(
            torch.tensor([1., earthquake_weight], dtype=torch.float, device=device)
        )
        self.train_info = []
        self.val_info = []
        self.start_time = time.time()
        self.best_val_loss = float('inf')

    def train_full(self, dataloader_train, phase='full', n_cycles=1, lr_decay=LR_DECAY):
        """Full training pass through all data"""
        self.current_phase = phase
        self.global_iter += 1
        print("Starting full training...")
        self.model.train()
        learning_rate = self.learning_rate

        for cycle in range(n_cycles):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            hid_state = self.model.init_state(batch_size=1, device=self.device)

            for i, (inputs, labels) in enumerate(dataloader_train):
                loss, iter_time = self._train_step(inputs, labels, hid_state, optimizer)

                # Logging
                if i % 100 == 0:
                    self._log_training(i, cycle, len(dataloader_train), loss, iter_time)

                # Validation
                if i % 250 == 0:
                    # self._validate(dataloader_train, i, cycle, len(dataloader_train), loss)
                    self._validate(dataloader_train, loss)

                if i % 10 == 0:
                    torch.cuda.empty_cache()

            learning_rate /= lr_decay
            print(f"Cycle {cycle + 1}/{n_cycles} completed")

        return self.best_val_loss

    def train_partial(self, dataset_train, phase='partial', n_cycles=N_CYCLES,
                      queue_length=QUEUE_LENGTH, lr_decay=LR_DECAY):
        """Partial training on random segments"""
        self.global_iter += 1
        self.current_phase = phase
        print("Starting partial training...")
        self.model.train()
        learning_rate = self.learning_rate

        for cycle in range(n_cycles):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            start = np.random.randint(0, len(dataset_train) - queue_length)
            hid_state = self.model.init_state(batch_size=BATCH_SIZE, device=self.device)

            for t in range(start, start + queue_length):
                inputs, labels = dataset_train[t]
                inputs = inputs.unsqueeze(0)
                labels = labels.unsqueeze(0)

                loss, _ = self._train_step(inputs, labels, hid_state, optimizer)

                if t % 10 == 0:
                    torch.cuda.empty_cache()

            learning_rate /= lr_decay
            print(f"Cycle {cycle + 1}/{n_cycles} completed")

    def _train_step(self, inputs, labels, hid_state, optimizer):
        """Single training step"""
        iter_start = time.time()

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()

        hid_state = tuple(h.detach() for h in hid_state)
        hid_state, outputs = self.model(inputs, hid_state)

        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        outputs = outputs.transpose(1, 2)
        labels = labels.view(-1)

        loss = self.criterion(outputs.view(-1, 2), labels)
        loss.backward()
        optimizer.step()

        return loss.item(), time.time() - iter_start

    def _log_training(self, i, cycle, dataloader_len, loss, iter_time):
        """Log training metrics"""
        self.train_info.append({
            "iter": self.global_iter,
            "loss": loss,
            "time": iter_time * 1000,
            "phase": self.current_phase  # Track training phase
        })
        self.global_iter += 1
        print(f"Cycle {cycle + 1}, Batch {i}, Loss: {loss:.4f}")

    @torch.no_grad()
    def _validate(self, dataloader, current_train_loss):
        """Perform validation with consistent batches"""
        self.model.eval()

        # Use fixed seed for consistent validation batches
        torch.manual_seed(42)

        total_loss = 0
        num_batches = 200

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                if i >= num_batches:
                    break

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                hid_state = self.model.init_state(batch_size=1, device=self.device)
                hid_state, outputs = self.model(inputs, hid_state)

                outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
                outputs = outputs.transpose(1, 2)
                labels = labels.view(-1)

                loss = self.criterion(outputs.view(-1, 2), labels)
                total_loss += loss.item()

        val_loss = total_loss / num_batches
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # Log validation info with global iteration
        self.val_info.append({
            "iter": self.global_iter,
            "train/loss": current_train_loss,
            "val/loss": val_loss,
            "phase": self.current_phase
        })

        self.model.train()
        return val_loss


class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader_test):
        """Evaluate model performance using ROC AUC and Average Precision scores"""
        predictions, targets = self._get_predictions(dataloader_test)

        # Calculate and return only the essential metrics
        return {
            'roc_auc': roc_auc_score(
                targets.reshape(-1).cpu(),
                predictions.reshape(-1).cpu()
            ),
            'avg_precision': average_precision_score(
                targets.reshape(-1).cpu(),
                predictions.reshape(-1).cpu()
            )
        }

    def _get_predictions(self, dataloader):
        """Get model predictions"""
        self.model.eval()
        predictions = []
        targets = []

        hid_state = self.model.init_state(batch_size=1, device=self.device)

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float()

            hid_state, outputs = self.model(inputs, hid_state)
            predictions.append(outputs[:, 1, :, :])
            targets.append(labels.squeeze(0))

        predictions = torch.stack(predictions)[10:]  # Cut initial predictions
        targets = torch.stack(targets)[10:]

        return predictions, targets


def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data and create datasets
    # Get the project root directory
    celled_data = torch.load(f"data/celled_data_{N_CELLS_HOR}x{N_CELLS_VER}", weights_only=True)
    freq_map = (celled_data > HEAVY_QUAKE_THRES).float().mean(dim=0).cpu()

    dataset_train = Dataset_RNN_Train(celled_data, device='cpu')
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=False,
                                  num_workers=0, pin_memory=True)

    # Initialize model and trainer
    model = LSTMCell(freq_map, device=device)
    trainer = Trainer(model, device)

    # Training phases
    print("Stage 1: Full training pass")
    best_val_loss = trainer.train_full(dataloader_train, n_cycles=1)

    print("Stage 2: Partial training")
    trainer.train_partial(dataset_train, n_cycles=N_CYCLES)

    print("Stage 3: Final full pass")
    trainer.train_full(dataloader_train, n_cycles=1)

    # Testing phase
    dataset_test = Dataset_RNN_Test(celled_data)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1)

    evaluator = Evaluator(model, device)
    test_results = evaluator.evaluate(dataloader_test)

    # Save results
    experiment_name = "earthquake_prediction"  # Or get this from args
    results_dict = {
        "final_train_loss": trainer.train_info[-1]["loss"],
        "best_val_loss": trainer.best_val_loss,
        "total_train_time": time.time() - trainer.start_time,
        "test_roc_auc": test_results['roc_auc'],
        "test_avg_precision": test_results['avg_precision']
    }

    # Structure for a single run
    formatted_results = {
        experiment_name: {
            "means": {
                f"{k}_mean": v for k, v in results_dict.items()
            },
            "stderrs": {
                f"{k}_stderr": 0.0 for k in results_dict.keys()  # Zero for single run
            },
            "final_info_dict": {
                k: [v] for k, v in results_dict.items()  # List with single value
            }
        }
    }

    all_results = {
        "earthquake_final_info": formatted_results,
        "earthquake_train_info": trainer.train_info,
        "earthquake_val_info": trainer.val_info,
    }

    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(formatted_results, f)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)


if __name__ == "__main__":
    main()
