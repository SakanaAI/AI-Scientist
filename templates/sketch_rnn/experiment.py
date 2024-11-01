# This file trains a Sketch RNN (https://arxiv.org/abs/1704.03477).

import argparse
import json
import os.path as osp
import pathlib
import pickle
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils


@dataclass
class State:
    """Probability distribution parameters for the next pen move."""
    mixture_logits: torch.Tensor
    mu_x: torch.Tensor
    mu_y: torch.Tensor
    sigma_x: torch.Tensor
    sigma_y: torch.Tensor
    rho_xy: torch.Tensor
    pen_logits: torch.Tensor


def sample_from_state(state, temperature, device):
    """Sample a pen move from the current state, and update the state."""
    # Sample a mixture.
    mixture_logits = state.mixture_logits.data.cpu().numpy()
    mixture_weights = utils.apply_temperature(mixture_logits, temperature)
    mixture_idx = np.random.choice(mixture_weights.size, p=mixture_weights)
    # Sample mixture params.
    mu_x = state.mu_x.data[mixture_idx].cpu()
    mu_y = state.mu_y.data[mixture_idx].cpu()
    sigma_x = state.sigma_x.data[mixture_idx].cpu()
    sigma_y = state.sigma_y.data[mixture_idx].cpu()
    rho_xy = state.rho_xy.data[mixture_idx].cpu()
    # Sample x, y with mixture.
    x, y = utils.sample_bivariate_normal(
        mu_x, mu_y, sigma_x, sigma_y, rho_xy, temperature,
    )
    # Sample pen state.
    pen_logits = state.pen_logits.data.cpu().numpy()
    pen_weights = utils.apply_temperature(pen_logits, temperature)
    pen_state_idx = np.random.choice(3, p=pen_weights)
    # Construct new decoder input state.
    next_state = torch.zeros(5)
    next_state[0] = x
    next_state[1] = y
    next_state[pen_state_idx + 2] = 1
    return (
        next_state.view(1, 1, -1).to(device),
        x, y, pen_state_idx == 1, pen_state_idx == 2,
    )


def compute_reconstruction_loss(state, targets):
    """Maximum likelihood of probability(target)."""
    num_mixtures = state.mu_x.size()[-1]
    dx = torch.stack([targets.data[:, :, 0]] * num_mixtures, dim=2)
    dy = torch.stack([targets.data[:, :, 1]] * num_mixtures, dim=2)
    pen_state = targets.data[:, :, 2:]
    mask = 1 - pen_state[:, :, -1]
    pdf_logits = utils.bivariate_normal_pdf(
        dx, dy,
        state.mu_x, state.mu_y, state.sigma_x, state.sigma_y, state.rho_xy
    )
    llh_xy = -torch.sum(
        mask * torch.logsumexp(state.mixture_logits + pdf_logits, dim=2)
    )
    llh_pen = -torch.sum(pen_state * state.pen_logits)
    return (llh_xy + llh_pen) / float(np.prod(mask.size()))


def compute_kl_loss(sigma, mu, kl_min):
    """KL between distribution of latent signals and IID N(0, I)."""
    kl_loss = -0.5 * (
        torch.sum(1 + sigma - mu ** 2 - torch.exp(sigma))
    ) / float(np.prod(sigma.size()))
    if kl_loss < kl_min:
        return kl_loss.detach()
    return kl_loss


class EncoderRNN(nn.Module):
    def __init__(self, config):
        super(EncoderRNN, self).__init__()
        self.hidden_size = config.encoder_hidden_size
        self.device = config.device
        # Bidirectional lstm:
        self.lstm = nn.LSTM(
            input_size=5,  # dx dy pen-down pen-up end.
            hidden_size=config.encoder_hidden_size,
            bidirectional=True
        )
        # Create mu and sigma from lstm's last output:
        self.fc_mu = nn.Linear(2 * self.hidden_size, config.latent_size)
        self.fc_sigma = nn.Linear(2 * self.hidden_size, config.latent_size)

    def forward(self, inputs, batch_size, hidden_cell_pair=None):
        if hidden_cell_pair is None:
            # Initialize with zeros.
            hidden = torch.zeros(2, batch_size, self.hidden_size)
            cell = torch.zeros(2, batch_size, self.hidden_size)
            hidden_cell_pair = (hidden.to(self.device), cell.to(self.device))

        _, (hidden, cell) = self.lstm(inputs.float(), hidden_cell_pair)
        # hidden is (2, batch_size, hidden_size),
        # we want it to be (batch_size, 2 * hidden_size):
        hidden_forward, hidden_backward = torch.split(hidden, 1, 0)
        hidden_forward_backward = torch.cat(
            [hidden_forward.squeeze(0), hidden_backward.squeeze(0)], dim=1
        )
        # mu and sigma:
        mu = self.fc_mu(hidden_forward_backward)
        sigma_hat = self.fc_sigma(hidden_forward_backward)
        sigma = torch.exp(sigma_hat / 2.)
        # noise ~ N(0, 1)
        noise = torch.normal(torch.zeros(mu.size()), torch.ones(mu.size()))
        latent_signal = mu + sigma * noise.to(self.device)
        # mu and sigma_hat are needed for kl loss
        return latent_signal, mu, sigma_hat


class DecoderRNN(nn.Module):
    def __init__(self, config):
        super(DecoderRNN, self).__init__()
        self.hidden_size = config.decoder_hidden_size
        # FC layer used to initialize hidden and cell from a latent signal:
        self.fc_hidden_cell = nn.Linear(
            config.latent_size, 2 * self.hidden_size
        )
        # Unidirectional lstm:
        self.lstm = nn.LSTM(
            input_size=config.latent_size + 5,
            hidden_size=config.decoder_hidden_size,
        )
        # FC that predict Mixture's parameters from hiddens activations.
        # The number of parameters is:
        # 5 * M (x and y means, x and y variances, xy covariances) 
        # + M (mixture weights) 
        # + 3 (pen-down, pen-up, end).
        self.num_params = 6 * config.num_mixtures + 3
        self.fc_mixture = nn.Linear(self.hidden_size, self.num_params)

    def forward(self, inputs, latent_signal, hidden_cell_pair=None):
        if hidden_cell_pair is None:
            # Initialize with latent signal.
            hidden_cell = F.tanh(self.fc_hidden_cell(latent_signal))
            hidden, cell = torch.split(hidden_cell, self.hidden_size, 1)
            # Remove unused first axis.
            hidden_cell_pair = (
                hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous()
            )
        outputs, (hidden, cell) = self.lstm(inputs, hidden_cell_pair)
        if self.training:
            # Teacher forcing mode: use whole output sequence.
            params = self.fc_mixture(outputs)
        else:
            # Inference mode: use last updated hidden signal.
            params = self.fc_mixture(hidden)
        # Separate pen and mixture params.
        params_sets = torch.split(params, 6, dim=-1)
        params_mixture = torch.stack(params_sets[:-1], dim=-1)  # Trajectory.
        pen_logits = params_sets[-1]  # Pen up/down + end.
        # Identify mixture params:
        mixture_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(
            params_mixture, 1, dim=2
        )
        return State(
            mixture_logits=F.log_softmax(mixture_logits.squeeze(), dim=-1),
            mu_x=mu_x.squeeze(),
            mu_y=mu_y.squeeze(),
            sigma_x=torch.exp(sigma_x.squeeze()),
            sigma_y=torch.exp(sigma_y.squeeze()),
            rho_xy=torch.tanh(rho_xy.squeeze()),
            pen_logits=F.log_softmax(pen_logits.squeeze(), dim=-1),
        ), hidden, cell


class Model():
    def __init__(self, config):
        self.device = config.device
        self.batch_size = config.batch_size
        self.grad_clip = config.grad_clip
        self.latent_size = config.latent_size
        self.sequence_length = config.sequence_length
        self.temperature = config.temperature
        # Build encoder and decoder and their optimizers.
        self.encoder = EncoderRNN(config).to(self.device)
        self.decoder = DecoderRNN(config).to(self.device)
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), config.learning_rate
        )
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), config.learning_rate
        )

        # Function to decay optimizers.
        def _decay(optimizer):
            for param_group in optimizer.param_groups:
                if param_group['lr'] > config.min_learning_rate:
                    param_group['lr'] *= config.learning_rate_decay_factor

        self.decay = _decay
        # kl loss parameters
        self.initial_kl_weight = config.initial_kl_weight
        self.kl_weight_decay = 1
        self.kl_weight_decay_factor = config.kl_weight_decay_factor
        self.kl_min = config.kl_min
        # eos and sos tokens:
        self.eos = (
            torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * config.batch_size)
        ).unsqueeze(0).to(self.device)
        self.sos = (
            torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * config.batch_size)
        ).unsqueeze(0).to(self.device)

    def train(self, sequences):
        self.encoder.train()
        self.decoder.train()
        # Encode sequences and update hidden variables mu and sigma.
        latent_signal, mu, sigma = self.encoder(sequences, self.batch_size)
        # Prepare decoder's input:
        # Put the sos token at the beggining.
        inputs = torch.cat([self.sos, sequences], 0)
        # Expend latent signal to be ready to concatenate with inputs.
        latent_signal_stack = torch.stack(
            [latent_signal] * (self.sequence_length + 1),
        )
        # Decoder input is concatenation of latent signal and sequence inputs.
        decoder_input = torch.cat([inputs, latent_signal_stack], dim=2)
        # Decode:
        state, _, _ = self.decoder(decoder_input, latent_signal)
        # Update kl weight.
        self.kl_weight = 1 - (1 - self.initial_kl_weight) * self.kl_weight_decay
        self.kl_weight_decay *= self.kl_weight_decay_factor
        # Compute losses.
        kl_loss = compute_kl_loss(mu, sigma, self.kl_min)
        targets = torch.cat([sequences, self.eos], dim=0).detach()
        reconstruction_loss = compute_reconstruction_loss(state, targets)
        loss = reconstruction_loss + self.kl_weight * kl_loss
        # Compute gradients.
        loss.backward()
        # Gradient cliping.
        nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)
        # Optimizers steps.
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        # Apply learning rate decay.
        self.decay(self.encoder_optimizer)
        self.decay(self.decoder_optimizer)
        # Flush optimizers.
        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)
        return (
            loss.detach().cpu().numpy(),
            reconstruction_loss.detach().cpu().numpy(),
            kl_loss.detach().cpu().numpy(),
        )

    def sample(self, context=None):
        """Samples a sequence of strokes."""
        self.encoder.eval()
        self.decoder.eval()
        if context is not None:
            # Condition generation with encoded context.
            latent_signal, _, _ = self.encoder(context, 1)
        else:
            latent_signal = torch.normal(
                torch.zeros(self.latent_size), torch.ones(self.latent_size)
            ).to(self.device).view(1, -1)
        sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).to(self.device)
        input = sos
        seq_x, seq_y, seq_z = [], [], []
        hidden_cell = None
        for _ in range(self.sequence_length):
            decoder_input = torch.cat(
                [input, latent_signal.unsqueeze(0)], dim=2
            )
            # Decode:
            state, hidden, cell = self.decoder(
                decoder_input, latent_signal, hidden_cell
            )
            hidden_cell = (hidden, cell)
            # Sample from parameters and update state.
            input, dx, dy, pen_down, eos = sample_from_state(
                state, self.temperature, self.device
            )
            # Append sampled stroke to generated sequence.
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                break
        # Visualize resulting sequence of strokes:
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        return np.stack([x_sample, y_sample, z_sample]).T


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sequence_length", type=int, default=100)
    parser.add_argument("--encoder_hidden_size", type=int, default=128)
    parser.add_argument("--decoder_hidden_size", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--num_mixtures", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--learning_rate_decay_factor", type=float, default=0.9999
    )
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--initial_kl_weight", type=float, default=0.01)
    parser.add_argument("--kl_weight_decay_factor", type=float, default=0.99999)
    parser.add_argument("--kl_min", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--out_dir", type=str, default="run_0")
    config = parser.parse_args()

    final_infos = {}
    all_results = {}

    pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)

    for dataset_name in ["cat", "butterfly", "yoga", "owl"]:

        # Prepare model
        if config.device == 'cuda':
            assert torch.cuda.is_available(), (
                "Device set to cuda, but cuda is unavailable."
            )
        model = Model(config)
        print("compiling the model... (takes a ~minute)")
        model.encoder = torch.compile(model.encoder)
        model.decoder = torch.compile(model.decoder)

        # Prepare data
        dataset = utils.get_dataset(dataset_name, config.sequence_length)
        get_batch = utils.get_batch_factory(
            dataset,
            config.sequence_length,
            config.device
        )

        # training loop
        train_reconstruction_losses = []
        train_kl_losses = []
        train_losses = []
        train_step_times = []
        for step in range(config.max_steps):
            batch = get_batch(config.batch_size)
            step_time = time.time()
            loss, reconstruction_loss, kl_loss = model.train(batch)
            train_step_time = time.time() - step_time
            train_reconstruction_losses.append(reconstruction_loss)
            train_kl_losses.append(kl_loss)
            train_losses.append(loss)
            train_step_times.append(train_step_time)
            if step % 100 == 0:
                print(
                    f'step {step}, loss {loss:.4f}',
                    f', recons. loss {reconstruction_loss:.4f}',
                    f', kl_loss {kl_loss:.4f}',
                    f', train_step_time {train_step_time:.4f}',
                )

        final_infos[dataset_name] = {
            "means": {
                "train_step_time": float(np.mean(train_step_times)),
                "loss": float(loss),
                "reconstruction_loss": float(reconstruction_loss),
                "kl_loss": float(kl_loss),
            }
        }

        # Save drawing.
        context = get_batch(1)
        all_results[dataset_name] = {
            "train_losses": train_losses,
            "train_reconstruction_losses": train_reconstruction_losses,
            "train_kl_losses": train_kl_losses,
            "conditioned_sequence": model.sample(context),
            "unconditioned_sequence": model.sample(),
        }

    with open(osp.join(config.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(osp.join(config.out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)
