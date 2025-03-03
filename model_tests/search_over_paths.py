import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import inspect

from model_visual2 import load_code, get_model_paths, load_model_architecture, get_denoising_steps, plot_denoising_process
from noise_search_2 import create_digit_dataloader, estimate_target_distribution

MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)

DEBUG_VISUALIZE_DIGIT = False  # Set to True to check dataset selection

def create_digit_dataloader(digit, batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    full_dataset = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    indices = [i for i, target in enumerate(full_dataset.targets) if target == digit]
    subset = Subset(full_dataset, indices)
    
    if DEBUG_VISUALIZE_DIGIT:
        sample_image, sample_label = full_dataset[indices[0]]
        plt.imshow(sample_image.squeeze(), cmap='gray')
        plt.title(f"Sample Digit: {sample_label}")
        plt.show()
    
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def estimate_target_distribution(model, digit, digit_loader, checkpoints, device="cuda"):
    sum_x0 = 0.0
    count = 0
    for batch in digit_loader:
        images, labels = batch
        assert (labels == digit).all(), "Digit loader contains incorrect labels!"
        imgs = images.to(device)
        sum_x0 += imgs.sum(dim=0)
        count += imgs.shape[0]
    global_mean = sum_x0 / count
    
    target_means = {}
    for t in checkpoints:
        factor = model.sqrt_alphas_cumprod[t]
        target_means[t] = factor * global_mean.unsqueeze(0)
    return target_means

def denoise_to_step(model, candidates, t, start_point, labels, device="cpu", use_clip=True):

    for step in range(start_point, t, -1):

        print(step)

        t_tensor = torch.full((candidates.shape[0],), step, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        if use_clip:
            candidates = model._reverse_diffusion_with_clip(candidates, t_tensor, noise, labels)
        else:
            candidates = model._reverse_diffusion(candidates, t_tensor, noise, labels)

    return candidates  # Denoised images at step `t`

def get_checkpoints(num_steps, delta_f, delta_b):

    checkpoints = []
    cur_step = num_steps - 1
    while cur_step - delta_b >= 0:
        cur_step -= delta_b
        checkpoints.append(cur_step)
        cur_step += delta_f

    checkpoints.append(0) # Step 0 is always a checkpoint

    return checkpoints

def lc_search_over_paths(n_candidates, delta_f, delta_b, model, model_ema, digit_to_generate, digit_loader, ema=False, n_samples=4, use_clip=True, device='cpu', scoring_method='mean_distribution_accuracy'):
    if delta_f > delta_b:
        raise ValueError("delta_f must be less than delta_b.")

    checkpoints = get_checkpoints(model.timesteps, delta_f, delta_b)

    if scoring_method == 'mean_distribution_accuracy':
        target_distribution = estimate_target_distribution(model, digit_to_generate, digit_loader, checkpoints, device=device)

    # Get first set of candidates
    candidates = torch.randn((n_candidates, model.in_channels, model.image_size, model.image_size), device=device)

    t = model.timesteps - 1
    while (t - delta_b) >= 0:
        noise = torch.randn_like(candidates, device=device)
        labels = torch.full((candidates.shape[0],), digit_to_generate, dtype=torch.long, device=device)

        # Denoise candidates to checkpoint
        candidates = denoise_to_step(model, candidates, t - delta_b, t, labels, device = device, use_clip=use_clip)

        t -= delta_b
        # Select top n candidates at checkpoint with scoring method
        if scoring_method == 'mean_distribution_accuracy':
            target = target_distribution[t].expand(candidates.shape[0], -1, -1, -1)
            scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
            k = min(candidates.shape[0], n_candidates)
            topk_indices = torch.topk(scores, k=k).indices
            candidates = candidates[topk_indices]

        print(f'Finished timestep {t}, kept {candidates.shape[0]} candidates.')

        # Expand each top candidate to n copies before renoising
        candidates = candidates.repeat_interleave(n_candidates, dim=0)  # Expands from n to n*10
        noise = torch.randn_like(candidates, device=device)  # Generate fresh noise for each expanded candidate

        # Renoise candidates
        candidates = model._partial_forward_diffusion(
            candidates,
            torch.full((candidates.shape[0],), t, dtype=torch.long, device=candidates.device),
            torch.full((candidates.shape[0],), t + delta_f, dtype=torch.long, device=candidates.device),
            noise
        )

        t += delta_f

    # Denoise candidates to step 0
    noise = torch.randn_like(candidates, device=device)
    labels = torch.full((candidates.shape[0],), digit_to_generate, dtype=torch.long, device=device)
    candidates = denoise_to_step(model, candidates, 0, t, labels, device = device, use_clip=use_clip)

    # Select best candidate
    if scoring_method == 'mean_distribution_accuracy':
        target = target_distribution[0].expand(candidates.shape[0], -1, -1, -1)
        scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
        best_idx = torch.argmax(scores)
        best_candidate = candidates[best_idx]

    for i in range(candidates.shape[0]):
        image = (candidates[i].squeeze().detach().cpu().numpy() + 1.0) / 2.0
        plt.imshow(image, cmap='gray')
        plt.title(f"Generated Image {i}")
        plt.show()

    best_image = (best_candidate.squeeze().detach().cpu().numpy() + 1.0) / 2.0
    plt.imshow(best_image, cmap='gray')
    plt.title("Best Generated Image")
    plt.show()

    return best_candidate

def singular_model_experiment():
    ### Parameters picked by user ###
    n_candidates = 3
    num_steps_per_checkpoint = 200
    digit_to_generate = 8
    delta_f = 30 # Number of steps to renoise back to
    delta_b = 60 # Number of steps to denoise back to

    # Search over paths
    ema = False
    n_samples = 1
    use_clip = True
    labels = None
    scoring_method = 'mean_distribution_accuracy'

    ### More generalized code ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = 'lc'  # 'lc' for label-conditioned, 'nlc' for non-label-conditioned

    # Load code
    TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage = load_code(model_type=model_type)

    # Load data
    train_loader, test_loader = create_mnist_dataloaders(batch_size=128,image_size=28)

    # Load model architecture
    model = load_model_architecture(MNISTDiffusion, device=device, model_type=model_type)
    model_ema = ExponentialMovingAverage(model, decay=0.995, device=device)

    # Get model filepaths
    sorted_model_paths, sorted_epoch_numbers = get_model_paths(TRAINED_MODELS_DIR)

    # Select model to load based on epoch number
    epoch_number = 100
    model_to_load = sorted_model_paths[sorted_epoch_numbers.index(epoch_number)]

    # Load model weights
    checkpoint = torch.load(model_to_load, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])

    digit_loader = create_digit_dataloader(digit=digit_to_generate, batch_size=128)

    lc_search_over_paths(n_candidates, delta_f, delta_b, model, model_ema, digit_to_generate, digit_loader, ema=ema, n_samples=n_samples, use_clip=use_clip, device=device, scoring_method=scoring_method)

if __name__ == '__main__':
    #scaling_experiment()
    singular_model_experiment()