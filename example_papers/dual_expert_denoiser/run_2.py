# This file trains a DDPM diffusion model on 2D datasets.

import argparse
import json
import time
import os.path as osp
import numpy as np
from tqdm.auto import tqdm
import npeet.entropy_estimators as ee
import pickle
import pathlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from ema_pytorch import EMA

import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_dim = self.dim // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_dim - 1)
        emb = torch.exp(-emb * torch.arange(half_dim)).to(device)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ff = nn.Linear(width, width)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return x + self.ff(self.act(x))


class MLPDenoiser(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 128,
            hidden_dim: int = 256,
            hidden_layers: int = 3,
    ):
        super().__init__()
        self.time_mlp = SinusoidalEmbedding(embedding_dim)
        # sinusoidal embeddings help capture high-frequency patterns for low-dim data
        self.input_mlp1 = SinusoidalEmbedding(embedding_dim, scale=25.0)
        self.input_mlp2 = SinusoidalEmbedding(embedding_dim, scale=25.0)

        self.gating_network = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.expert1 = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            *[ResidualBlock(hidden_dim) for _ in range(hidden_layers)],
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.expert2 = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            *[ResidualBlock(hidden_dim) for _ in range(hidden_layers)],
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        emb = torch.cat([x1_emb, x2_emb, t_emb], dim=-1)
        
        gating_weight = self.gating_network(emb)
        expert1_output = self.expert1(emb)
        expert2_output = self.expert2(emb)
        
        return gating_weight * expert1_output + (1 - gating_weight) * expert2_output


class NoiseScheduler():
    def __init__(
            self,
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
    ):
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        elif beta_schedule == "quadratic":
            self.betas = (torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2).to(device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.).to(device)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = (self.alphas_cumprod ** 0.5).to(device)
        self.sqrt_one_minus_alphas_cumprod = ((1 - self.alphas_cumprod) ** 0.5).to(device)

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod).to(device)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1).to(device)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod).to(
            device)
        self.posterior_mean_coef2 = ((1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)).to(device)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_timesteps", type=int, default=100)
    parser.add_argument("--num_train_steps", type=int, default=10000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="run_0")
    config = parser.parse_args()

    final_infos = {}
    all_results = {}

    pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)

    for dataset_name in ["circle", "dino", "line", "moons"]:
        dataset = datasets.get_dataset(dataset_name, n=100000)
        dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

        model = MLPDenoiser(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_size,
            hidden_layers=config.hidden_layers,
        ).to(device)
        ema_model = EMA(model, beta=0.995, update_every=10).to(device)

        noise_scheduler = NoiseScheduler(num_timesteps=config.num_timesteps, beta_schedule=config.beta_schedule)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=config.num_train_steps)
        train_losses = []
        print("Training model...")

        model.train()
        global_step = 0
        progress_bar = tqdm(total=config.num_train_steps)
        progress_bar.set_description("Training")

        start_time = time.time()
        while global_step < config.num_train_steps:
            for batch in dataloader:
                if global_step >= config.num_train_steps:
                    break
                batch = batch[0].to(device)
                noise = torch.randn(batch.shape).to(device)
                timesteps = torch.randint(
                    0, noise_scheduler.num_timesteps, (batch.shape[0],)
                ).long().to(device)

                noisy = noise_scheduler.add_noise(batch, noise, timesteps)
                noise_pred = model(noisy, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
                ema_model.update()

                scheduler.step()
                progress_bar.update(1)
                logs = {"loss": loss.detach().item()}
                train_losses.append(loss.detach().item())
                progress_bar.set_postfix(**logs)
                global_step += 1

        progress_bar.close()
        end_time = time.time()
        training_time = end_time - start_time

        # Eval loss
        model.eval()
        eval_losses = []
        for batch in dataloader:
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long().to(device)
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            eval_losses.append(loss.detach().item())
        eval_loss = np.mean(eval_losses)

        # Eval image saving
        ema_model.eval()
        sample = torch.randn(config.eval_batch_size, 2).to(device)
        timesteps = list(range(len(noise_scheduler)))[::-1]
        inference_start_time = time.time()
        for t in timesteps:
            t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long().to(device)
            with torch.no_grad():
                residual = ema_model(sample, t)
            sample = noise_scheduler.step(residual, t[0], sample)
        sample = sample.cpu().numpy()
        inference_end_time = time.time()
        inference_time = inference_end_time - inference_start_time

        # Eval estimated KL
        real_data = dataset.tensors[0].numpy()
        kl_divergence = ee.kldiv(real_data, sample, k=5)

        # Calculate gating weights for visualization
        with torch.no_grad():
            x = torch.from_numpy(sample).float().to(device)
            t = torch.zeros(x.shape[0], dtype=torch.long).to(device)
            gating_weights = ema_model.ema_model.gating_network(
                torch.cat([
                    ema_model.ema_model.input_mlp1(x[:, 0]),
                    ema_model.ema_model.input_mlp2(x[:, 1]),
                    ema_model.ema_model.time_mlp(t)
                ], dim=-1)
            ).cpu().numpy()

        final_infos[dataset_name] = {
            "means": {
                "training_time": training_time,
                "eval_loss": eval_loss,
                "inference_time": inference_time,
                "kl_divergence": kl_divergence,
            }
        }

        all_results[dataset_name] = {
            "train_losses": train_losses,
            "images": sample,
            "gating_weights": gating_weights,
        }

    with open(osp.join(config.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(osp.join(config.out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)
