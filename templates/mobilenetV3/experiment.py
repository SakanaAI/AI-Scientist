import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# _make_divisible function from torchvision
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function ensures that all layers have a channel number that is divisible by 8.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that rounding down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# Squeeze-and-Excitation block
class SqueezeExcitation(nn.Module):
    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            activation: Callable[..., nn.Module] = nn.ReLU,
            scale_activation: Callable[..., nn.Module] = nn.Hardsigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation(inplace=True)
        self.scale_activation = scale_activation(inplace=True)

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input)
        return input * scale


# ConvNormActivation block
class ConvNormActivation(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]] = 3,
            stride: Union[int, Tuple[int]] = 1,
            padding: Optional[Union[int, Tuple[int], str]] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
            dilation: Union[int, Tuple[int]] = 1,
            bias: Optional[bool] = None,
    ) -> None:

        if padding is None:
            if isinstance(kernel_size, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                padding = tuple((k - 1) // 2 * d for k, d in zip(kernel_size, dilation))
        if bias is None:
            bias = norm_layer is None

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)
        self.out_channels = out_channels


# InvertedResidualConfig class
class InvertedResidualConfig:
    def __init__(
            self,
            input_channels: int,
            kernel: int,
            expanded_channels: int,
            out_channels: int,
            use_se: bool,
            activation: str,
            stride: int,
            dilation: int,
            width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.activation = activation
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


# InvertedResidual block
class InvertedResidual(nn.Module):
    def __init__(
            self,
            cnf: InvertedResidualConfig,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("Illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.activation == "HS" else nn.ReLU

        # Expand phase
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # Depthwise convolution
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                dilation=cnf.dilation,
            )
        )

        # Squeeze-and-Excitation
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(
                se_layer(
                    cnf.expanded_channels,
                    squeeze_channels,
                    activation=nn.ReLU,
                )
            )

        # Project phase
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self.is_strided = cnf.stride > 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            return input + result
        else:
            return result


# MobileNetV3 Small model
class MobileNetV3Small(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            dropout: float = 0.2,
            reduced_tail: bool = False,
            dilated: bool = False,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)

        # Build inverted residual setting
        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1

        inverted_residual_setting = [
            # input_c, kernel, exp_c, out_c, se, nl, s, d
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288 // reduce_divider, 96 // reduce_divider, True, "HS", 2, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]

        last_channel = _make_divisible(1024 // reduce_divider * width_mult, 8)

        # First layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # Building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(InvertedResidual(cnf, norm_layer))

        # Building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = _make_divisible(576 * width_mult, 8)
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Function to create the model and load pretrained weights
def mobilenet_v3_small(pretrained=False, progress=True, **kwargs):
    model = MobileNetV3Small(**kwargs)

    if pretrained:
        # Load the torchvision model with pretrained weights
        from torchvision.models import mobilenet_v3_small as tv_mobilenet_v3_small
        from torchvision.models import MobileNet_V3_Small_Weights

        # Check for number of classes
        if kwargs.get('num_classes', 1000) != 1000:
            # We cannot load the classifier weights (different classes)
            pretrained_model = tv_mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT, progress=progress)
            pretrained_state_dict = pretrained_model.state_dict()
            # Remove classifier weights
            pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if not k.startswith('classifier')}
            model_dict = model.state_dict()
            print(model_dict.keys())
            # Update the model dict
            model_dict.update(pretrained_state_dict)
            model.load_state_dict(model_dict)
        else:
            # Load all weights
            pretrained_model = tv_mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT, progress=progress)
            model.load_state_dict(pretrained_model.state_dict())

    return model


@dataclass
class Config:
    # data
    data_path: str = './data'
    dataset: str = 'cifar10'
    num_classes: int = 10
    # model
    model: str = 'mobilenet_v3_small'
    # training
    batch_size: int = 128
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    epochs: int = 2
    # system
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 2
    # logging
    log_interval: int = 100
    eval_interval: int = 1000
    # output
    out_dir: str = 'run_0'
    seed: int = 0
    # compile for SPEED!
    compile_model: bool = False


def get_data_loaders(config):
    if config.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(root=config.data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_test)
    elif config.dataset == 'cifar100':
        # Placeholder for CIFAR-100 (for future use)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        train_dataset = datasets.CIFAR100(root=config.data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_test)
        config.num_classes = 100  # Update number of classes for CIFAR-100
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


def train(config):
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed_all(config.seed)

    model = mobilenet_v3_small(pretrained=False, progress=True, num_classes=config.num_classes).to(config.device)

    if config.compile_model:
        print("Compiling the model...")
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    train_loader, test_loader = get_data_loaders(config)

    best_acc = 0.0
    train_log_info = []
    val_log_info = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            if batch_idx % config.log_interval == 0:
                train_log_info.append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * train_correct / train_total,
                    'lr': optimizer.param_groups[0]['lr']
                })
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {train_loss / (batch_idx + 1):.3f}, '
                      f'Acc: {100. * train_correct / train_total:.3f}%, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        val_loss, val_acc = evaluate(model, test_loader, criterion, config)
        val_log_info.append({
            'epoch': epoch,
            'loss': val_loss,
            'acc': val_acc
        })
        print(f'Validation - Loss: {val_loss:.3f}, Acc: {val_acc:.3f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.out_dir, 'best_model.pth'))

        scheduler.step()

    return train_log_info, val_log_info, best_acc


def evaluate(model, dataloader, criterion, config):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(dataloader)
    val_acc = 100. * val_correct / val_total

    return val_loss, val_acc


def test(config):
    model = MobileNetV3Small(num_classes=config.num_classes).to(config.device)
    if config.compile_model:
        print("Compiling the model for testing...")
        model = torch.compile(model)
    model.load_state_dict(torch.load(os.path.join(config.out_dir, 'best_model.pth')))
    _, test_loader = get_data_loaders(config)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(model, test_loader, criterion, config)
    print(f'Test - Loss: {test_loss:.3f}, Acc: {test_acc:.3f}%')
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV3 for Image Classification")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to save/load the dataset")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Outputs will be saved to {args.out_dir}")

    # Define datasets and number of seeds per dataset
    datasets = ['cifar10']  # For now, only CIFAR-10; can add 'cifar100' in the future
    num_seeds = {
        'cifar10': 1  # Change the number of seeds as desired
    }

    all_results = {}
    final_infos = {}

    for dataset in datasets:
        final_info_list = []
        for seed_offset in range(num_seeds[dataset]):
            # Update the config for each run
            config = Config(
                data_path=args.data_path,
                dataset=dataset,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                out_dir=args.out_dir,
                seed=seed_offset  # Set the seed
            )
            os.makedirs(config.out_dir, exist_ok=True)
            print(f"Starting training for {dataset} with seed {seed_offset}")
            start_time = time.time()
            train_log_info, val_log_info, best_acc = train(config)
            total_time = time.time() - start_time

            # Run test after training
            test_loss, test_acc = test(config)

            # Prepare final_info dictionary
            final_info = {
                "best_val_acc": best_acc,
                "test_acc": test_acc,
                "total_train_time": total_time,
                "config": vars(config)
            }
            final_info_list.append(final_info)

            # Store results in all_results
            key_prefix = f"{dataset}_{seed_offset}"
            all_results[f"{key_prefix}_final_info"] = final_info
            all_results[f"{key_prefix}_train_log_info"] = train_log_info
            all_results[f"{key_prefix}_val_log_info"] = val_log_info

            print(f"Training completed for {dataset} seed {seed_offset}. Best validation accuracy: {best_acc:.2f}%, Test accuracy: {test_acc:.2f}%")

        # Aggregate results over seeds
        final_info_dict = {k: [d[k] for d in final_info_list if k in d] for k in final_info_list[0].keys()}
        means = {f"{k}_mean": np.mean(v) for k, v in final_info_dict.items() if isinstance(v[0], (int, float, float))}
        stderrs = {f"{k}_stderr": np.std(v) / np.sqrt(len(v)) for k, v in final_info_dict.items() if isinstance(v[0], (int, float, float))}
        final_infos[dataset] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict
        }

    # Save final_infos to final_info.json
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f, indent=2)

    # Save all_results to all_results.npy
    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)

    print(f"All results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
