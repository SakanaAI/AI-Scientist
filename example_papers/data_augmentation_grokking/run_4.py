import argparse
import abc
import random
from itertools import permutations
from typing import Set
import os
import json
import numpy as np
from einops import rearrange, repeat
import torch
from torch.utils.data import IterableDataset
from torch import nn, Tensor


class AbstractDataset(abc.ABC):
    def __init__(self, group_elements1: Set, group_elements2: Set, frac_train: float):
        self.frac_train = frac_train
        self.group_elements1 = group_elements1
        self.group_elements2 = group_elements2
        self.ordered_group_elements1 = list(self.group_elements1)
        self.ordered_group_elements2 = list(self.group_elements2)
        self.idx2vocab = ["o", "="] + list(group_elements1.union(group_elements2))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        self.n_out = len(group_elements1.union(group_elements2))
        idxs = list(range(len(self.group_elements1) * len(self.group_elements2)))
        random.shuffle(idxs)
        self.train_pairs, self.val_pairs = (
            idxs[: int(len(idxs) * frac_train)],
            idxs[int(len(idxs) * frac_train) :],
        )

    @abc.abstractmethod
    def fetch_output(self, a, b):
        pass

    def encode(self, sequence):
        return [self.vocab2idx[item] for item in sequence]

    def decode(self, sequence):
        return [self.idx2vocab[item] for item in sequence]

    def form_equation(self, a, b, c):
        return [a, "o", b, "=", c]

    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def fetch_train_example(self):
        idx = random.choice(self.train_pairs)
        return self.fetch_example(idx)

    def fetch_val_example(self):
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)

    def reverse_operands(self, a, b):
        return b, a


class ModSumDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModSumDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p

    def fetch_output(self, a, b):
        return (a + b) % self.p

    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        if random.random() < 0.3:
            a, b = self.reverse_operands(a, b)
        if random.random() < 0.3:
            a, b = self.negate_operands(a, b)
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def negate_operands(self, a, b):
        return (self.p - a) % self.p, (self.p - b) % self.p


class ModSubtractDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModSubtractDataset, self).__init__(
            set(range(p)), set(range(p)), frac_train
        )
        self.p = p

    def fetch_output(self, a, b):
        return (a - b) % self.p

    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        rand = random.random()
        if rand < 0.15:
            a, b = self.reverse_operands(a, b)
        elif rand < 0.3:
            a, b = self.negate_operands(a, b)
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def reverse_operands(self, a, b):
        return b, a

    def negate_operands(self, a, b):
        return (self.p - a) % self.p, (self.p - b) % self.p


class ModDivisonDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModDivisonDataset, self).__init__(
            set(range(p)), set(range(1, p)), frac_train
        )
        self.p = p

    def fetch_output(self, a, b):
        return (a * pow(b, self.p - 2, self.p)) % self.p

    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        if random.random() < 0.3:
            a, b = self.negate_operands(a, b)
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c] - 2), equation

    def negate_operands(self, a, b):
        return (self.p - a) % self.p, b  # Only negate the dividend


class PermutationGroup(AbstractDataset):
    def __init__(self, k, frac_train):
        perms = set(map(tuple, permutations(list(range(k)))))
        super(PermutationGroup, self).__init__(perms, perms, frac_train)
        self.k = k

    def fetch_output(self, a, b):
        return tuple([a[b[i]] for i in range(len(b))])


class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {"train", "val"}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == "train":
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == "val":
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        x, y, _ = self.fetch_f()
        return torch.tensor(x), torch.tensor(y)


def operation_mod_p_data(operation: str, p: int, frac_train: float):
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    if operation == "x_plus_y":
        data = ModSumDataset(p=p, frac_train=frac_train)
    elif operation == "x_minus_y":
        data = ModSubtractDataset(p=p, frac_train=frac_train)
    elif operation == "x_div_y":
        data = ModDivisonDataset(p=p, frac_train=frac_train)
    elif operation == "permutation":
        data = PermutationGroup(k=5, frac_train=frac_train)
    return data


def get_data(operation: str, prime: int, training_fraction: float, batch_size: int):
    dataset = operation_mod_p_data(operation, prime, training_fraction)
    train_dataset = GroupDataset(dataset, "train")
    val_dataset = GroupDataset(dataset, "val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return (
        train_loader,
        val_loader,
        train_dataset.dataset.n_vocab,
        train_dataset.dataset.n_out,
    )


class DecoderBlock(torch.nn.Module):
    def __init__(self, dim_model: int, n_heads: int):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
        self.self_attn_norm = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 4),
            nn.GELU(),
            nn.Linear(dim_model * 4, dim_model),
        )
        self.ffn_norm = nn.LayerNorm(dim_model)

    def forward(self, x: Tensor):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        a1 = self.self_attn_norm(x + a1)
        a2 = self.ffn(a1)
        a2 = self.ffn_norm(a1 + a2)

        return a2


class Transformer(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim_model: int,
        num_heads: int,
        vocab_size: int,
        output_size: int,
        seq_len: int,
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, dim_model)
        self.position_embeddings = nn.Embedding(seq_len, dim_model)
        self.model = nn.Sequential(
            *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, output_size),
        )

    def forward(self, inputs: Tensor):
        batch_size, context_len = inputs.shape

        token_embedding = self.token_embeddings(inputs)

        positions = repeat(
            torch.arange(context_len, device=inputs.device), "p -> b p", b=batch_size
        )
        position_embedding = self.position_embeddings(positions)

        embedding = token_embedding + position_embedding

        embedding = rearrange(embedding, "b s d -> s b d")

        return self.model(embedding)


def train(model, train_loader, val_loader, optimizer, scheduler, device, num_train_batches, num_eval_batches):
    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    loss_total, correct = 0.0, 0.0
    total = 0

    step_val_acc_95 = None
    prev_val_acc = 0
    max_acc_increase_rate = 0

    # Loop over each batch from the training set
    count = 0
    for batch in train_loader:
        count += 1
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)[-1, :, :]
        loss = criterion(output, labels)
        correct += (torch.argmax(output, dim=1) == labels).sum()
        loss_total += loss * len(labels)
        total += len(labels)
        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        scheduler.step()

        # Evaluate on validation set
        if count % 100 == 0:
            val_metrics = evaluate(model, val_loader, device, num_eval_batches)
            val_acc = val_metrics["val_accuracy"]
            
            # Check for 95% validation accuracy
            if step_val_acc_95 is None and val_acc >= 0.95:
                step_val_acc_95 = count * num_train_batches

            # Calculate rate of validation accuracy increase
            acc_increase_rate = (val_acc - prev_val_acc) / 100
            max_acc_increase_rate = max(max_acc_increase_rate, acc_increase_rate)
            prev_val_acc = val_acc

        if count >= num_train_batches:
            break

    acc = correct / total
    loss = loss_total / total

    metrics = {
        "train_accuracy": float(acc),
        "train_loss": float(loss),
        "step_val_acc_95": step_val_acc_95,
        "max_acc_increase_rate": max_acc_increase_rate,
    }
    return metrics


def evaluate(model, val_loader, device, num_eval_batches):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.0
    total = 0
    count = 0
    # Loop over each batch from the validation set
    for batch in val_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Forward pass
        with torch.no_grad():
            output = model(inputs)[-1, :, :]
            correct += (torch.argmax(output, dim=1) == labels).sum()
            loss += criterion(output, labels) * len(labels)
            total += labels.shape[0]
        count += 1
        if count >= num_eval_batches:
            break

    acc = correct / total
    loss = loss / total

    metrics = {"val_accuracy": float(acc), "val_loss": float(loss)}
    return metrics


def run(out_dir, dataset, seed_offset):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337 + seed_offset)
    train_loader, val_loader, n_vocab, n_output = get_data(
        operation=dataset,
        prime=97,
        training_fraction=0.5,
        batch_size=512,
    )

    model = Transformer(
        num_layers=2,
        dim_model=128,
        num_heads=4,
        vocab_size=n_vocab,
        output_size=n_output,
        seq_len=5,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.98),
        weight_decay=0.5,
    )
    num_train_batches = 10
    num_eval_batches = 8
    num_total_updates = 7500
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: min(s / warmup_steps, 1)
    )

    final_info, train_log_info, val_log_info = [], [], []
    step_val_acc_99 = num_total_updates
    for ep in range(num_total_updates // num_train_batches):
        train_metrics = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            num_train_batches,
            num_eval_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            num_eval_batches,
        )
        train_metrics["step"] = (ep + 1) * num_train_batches
        val_metrics["step"] = (ep + 1) * num_train_batches

        if step_val_acc_99 == num_total_updates and val_metrics["val_accuracy"] > 0.99:
            step_val_acc_99 = val_metrics["step"]
        train_log_info.append(train_metrics)
        val_log_info.append(val_metrics)

    final_info = {
        "final_train_loss": train_metrics["train_loss"],
        "final_val_loss": val_metrics["val_loss"],
        "final_train_acc": train_metrics["train_accuracy"],
        "final_val_acc": val_metrics["val_accuracy"],
        "step_val_acc_99": step_val_acc_99 if step_val_acc_99 != num_total_updates else None,
        "step_val_acc_95": train_metrics["step_val_acc_95"],
        "max_acc_increase_rate": train_metrics["max_acc_increase_rate"],
    }
    print(final_info)
    with open(
        os.path.join(out_dir, f"final_info_{dataset}_{seed_offset}.json"), "w"
    ) as f:
        json.dump(final_info, f)
    return final_info, train_log_info, val_log_info


parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()


if __name__ == "__main__":
    num_seeds = {
        "x_div_y": 3,
        "x_plus_y": 3,
        "x_minus_y": 3,
        "permutation": 3,
    }

    out_dir = args.out_dir
    all_results = {}
    final_infos = {}
    for dataset in ["x_div_y", "x_minus_y", "x_plus_y", "permutation"]:
        final_info_list = []
        for seed_offset in range(num_seeds[dataset]):
            print(f"Running {dataset} with seed offset {seed_offset}")
            final_info, train_info, val_info = run(args.out_dir, dataset, seed_offset)
            all_results[f"{dataset}_{seed_offset}_final_info"] = final_info
            all_results[f"{dataset}_{seed_offset}_train_info"] = train_info
            all_results[f"{dataset}_{seed_offset}_val_info"] = val_info
            final_info_list.append(final_info)
        final_info_dict = {
            k: [d[k] for d in final_info_list if d[k] is not None] for k in final_info_list[0].keys()
        }
        means = {f"{k}_mean": np.mean(v) if v else None for k, v in final_info_dict.items()}
        stderrs = {
            f"{k}_stderr": np.std(v) / np.sqrt(len(v)) if v else None for k, v in final_info_dict.items()
        }
        final_infos[dataset] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict,
        }

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)
