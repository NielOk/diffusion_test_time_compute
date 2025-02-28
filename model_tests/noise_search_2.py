import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Constants
CHECKPOINT = "epoch_100_steps_00046900.pt"
DIGIT = 2  # Change this to any digit 0-9
SCORING_TIMESTEPS = 200
N_CANDIDATES = 16  # Number of noise samples for reverse diffusion
CHECKPOINT_INTERVAL = SCORING_TIMESTEPS  # Control pruning frequency
N_EXPERIMENTS = 2  # Number of times the experiment runs
DEBUG_VISUALIZE_DIGIT = True  # Set to True to check dataset selection

# Directories and model import
MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'nlc_gpu_accelerated_training')
TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'nlc_trained_ddpm', 'results')
sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
sys.path.append(TRAINED_MODELS_DIR)
from model import MNISTDiffusion  # Adjust import path if needed

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

def load_model(ckpt_path, device="cuda"):
    model = MNISTDiffusion(
        timesteps=1000,
        image_size=28,
        in_channels=1,
        base_dim=64,
        dim_mults=[2, 4]
    )
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model

def estimate_target_distribution(model, digit_loader, checkpoints, device="cuda"):
    sum_x0 = 0.0
    count = 0
    for batch in digit_loader:
        images, labels = batch
        assert (labels == DIGIT).all(), "Digit loader contains incorrect labels!"
        imgs = images.to(device)
        sum_x0 += imgs.sum(dim=0)
        count += imgs.shape[0]
    global_mean = sum_x0 / count
    
    target_means = {}
    for t in checkpoints:
        factor = model.sqrt_alphas_cumprod[t]
        target_means[t] = factor * global_mean.unsqueeze(0)
    return target_means

def reverse_diffusion_search(model, target_means, n_candidates=N_CANDIDATES, checkpoint_interval=CHECKPOINT_INTERVAL, device="cuda"):
    candidates = torch.randn((n_candidates, model.in_channels, model.image_size, model.image_size), device=device)
    
    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)
        
        if t % checkpoint_interval == 0 and t in target_means:
            target = target_means[t].expand(candidates.shape[0], -1, -1, -1)
            scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
            k = max(1, candidates.shape[0] // 2)
            topk_indices = torch.topk(scores, k=k).indices
            candidates = candidates[topk_indices]
            print(f"At timestep {t}, kept {candidates.shape[0]} candidates.")
    
    if 0 in target_means:
        target = target_means[0].expand(candidates.shape[0], -1, -1, -1)
        final_scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
        best_idx = torch.argmax(final_scores)
        best_candidate = candidates[best_idx]
    else:
        best_candidate = candidates[0]
    return best_candidate

def plot_image(image_tensor, title="Generated Image"):
    image = (image_tensor.squeeze().detach().cpu().numpy() + 1) / 2.0
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, CHECKPOINT)
    model = load_model(ckpt_path, device)
    
    digit_loader = create_digit_dataloader(digit=DIGIT, batch_size=128)
    checkpoints = list(range(0, model.timesteps, SCORING_TIMESTEPS))
    print(f"Using checkpoints: {checkpoints}")
    
    target_means = estimate_target_distribution(model, digit_loader, checkpoints, device)
    outputs = []
    for i in range(N_EXPERIMENTS):
        best_noise = reverse_diffusion_search(model, target_means, device=device)
        outputs.append(best_noise)
        print(f"Experiment {i+1} complete.")
    
    num_cols = 5
    num_rows = (N_EXPERIMENTS + num_cols - 1) // num_cols
    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    for idx, out in enumerate(outputs):
        plt.subplot(num_rows, num_cols, idx+1)
        plot_image(out, title=f"Run {idx+1}")
    plt.suptitle(f"Reverse Diffusion Outputs for Digit {DIGIT}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


