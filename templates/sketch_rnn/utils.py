import numpy as np
import torch


####################### probabilities utils

def bivariate_normal_pdf(dx, dy, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    """Probability density function of a (dx, dy) pair."""
    z_x = ((dx - mu_x) / sigma_x) ** 2
    z_y = ((dy - mu_y) / sigma_y) ** 2
    z_xy = (dx - mu_x) * (dy - mu_y) / (sigma_x * sigma_y)
    z = z_x + z_y - 2 * rho_xy * z_xy
    logits = -z / (2 * (1 - rho_xy ** 2))
    norm = (2 * np.pi * sigma_x * sigma_y) * torch.sqrt(1 - rho_xy ** 2)
    return logits - torch.log(norm)


def sample_bivariate_normal(
        mu_x, mu_y, sigma_x, sigma_y, rho_xy, temperature, greedy=False,
):
    """Samples from bivariate normal distribution."""
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]
    sigma_x *= np.sqrt(temperature)
    sigma_y *= np.sqrt(temperature)
    cov = [
        [sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
        [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]
    ]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def apply_temperature(logits, temperature):
    """Adjusts logits to get a softmax with a given temperature."""
    adjusted_logits = logits / temperature
    adjusted_logits -= np.max(adjusted_logits)
    exp_adjusted_logits = np.exp(adjusted_logits)
    return exp_adjusted_logits / np.sum(exp_adjusted_logits)


####################### dataset utils


def purify(strokes, sequence_length):
    """Removes to small or too long sequences + removes large gaps."""
    data = []
    for seq in strokes:
        if seq.shape[0] <= sequence_length and seq.shape[0] > 10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data


def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)


def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data


def get_dataset(dataset_name, sequence_length):
    file_name = 'datasets/' + dataset_name + '.npz'
    dataset = np.load(file_name, encoding='latin1', allow_pickle=True)['train']
    dataset = purify(dataset, sequence_length)
    dataset = normalize(dataset)
    return dataset


def get_batch_factory(dataset, sequence_length, device):
    """Defines a function that samples batch."""

    def get_batch(batch_size):
        """Samples a batch"""
        batch_idx = np.random.choice(len(dataset), batch_size)
        batch_sequences = [dataset[idx] for idx in batch_idx]
        strokes = []
        indice = 0
        for seq in batch_sequences:
            len_seq = len(seq[:, 0])
            new_seq = np.zeros((sequence_length, 5))
            new_seq[:len_seq, :2] = seq[:, :2]
            new_seq[:len_seq - 1, 2] = 1 - seq[:-1, 2]
            new_seq[:len_seq, 3] = seq[:, 2]
            new_seq[len_seq - 1:, 4] = 1
            new_seq[len_seq - 1, 2:4] = 0
            strokes.append(new_seq)
            indice += 1

        return torch.from_numpy(np.stack(strokes, 1)).float().to(device)

    return get_batch
