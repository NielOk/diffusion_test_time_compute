import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt

from greyscale_dynamic_forward_diffuser import GreyscaleDynamicForwardDiffuser

def cosine_beta_schedule(T, s=0.008):
    t = np.linspace(0, T, T+1)
    f = np.cos((t / T + s) / (1 + s) * (np.pi / 2))**2
    alphas_bar = f / f[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return np.clip(betas, 0.0001, 0.9999)  # Ensure numerical stability

def run_mnist_tests():

    data_dir = './data'
    
    ForwardDiffuser = GreyscaleDynamicForwardDiffuser()

    ForwardDiffuser.load_training_data(data_dir, download=True)

    # Create batches
    batch_size = 10
    data_batches, label_batches = ForwardDiffuser.create_batches(ForwardDiffuser.non_noisy_data, ForwardDiffuser.non_noisy_labels, batch_size, shuffle=True)

    T = 10 # num diffusion steps
    betas = cosine_beta_schedule(T, s=0.008)

    cur_batch = data_batches[0]
    cur_labels = label_batches[0]

    print(cur_labels)

    # Forward diffusion on inputs to get noisy data. The prev_step_noise of step 1 is prev_step_noise[1], is noise added at step 0
    batch_steps, prev_step_noises = ForwardDiffuser.batch_beta_schedule_forward_diffusion(cur_batch, T, betas)

    for step, batch in batch_steps.items():
        first_row = batch[1]
        plt.imshow(first_row, cmap='gray')
        plt.show()

    # Get the embeddings and normalize the batch steps
    batch_steps = ForwardDiffuser.embed_and_normalize(batch_steps, cur_labels, T)

if __name__ == '__main__':
    run_mnist_tests()