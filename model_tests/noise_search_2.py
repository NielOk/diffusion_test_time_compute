# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import sys
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset

# # Directories and model import
# MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
# GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'gpu_accelerated_training')
# TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'trained_ddpm', 'results')
# sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
# sys.path.append(TRAINED_MODELS_DIR)
# from model import MNISTDiffusion  # Adjust import path if needed

# # Create MNIST dataloader for all digits (used for training)...
# def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):
#     preprocess = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     train_dataset = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
#     return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# # Create a dataloader that only returns digit '1'
# def create_digit_dataloader(digit, batch_size, image_size=28, num_workers=4):
#     preprocess = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])
#     full_dataset = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
#     # Find indices where label equals the desired digit
#     indices = [i for i, target in enumerate(full_dataset.targets) if target == digit]
#     subset = Subset(full_dataset, indices)
#     return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# def load_model(ckpt_path, device="cuda"):
#     model = MNISTDiffusion(
#         timesteps=1000,
#         image_size=28,
#         in_channels=1,
#         base_dim=64,
#         dim_mults=[2, 4]
#     )
#     model.to(device)
#     ckpt = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model"], strict=False)
#     model.eval()
#     return model

# #######################################
# # NEW: Estimate target forward diffusion distribution for digit 1
# #######################################
# def estimate_target_distribution(model, digit_loader, checkpoints, device="cuda"):
#     """
#     Computes the target distribution q(x_t|y=1) for a set of checkpoint timesteps.
#     Since for the forward process the closed-form mean is:
#         x_t = sqrt(alphas_cumprod[t]) * x0   (with noise mean zero),
#     we estimate the mean of digit 1 images and then compute for each checkpoint:
#         target_mean[t] = sqrt(alphas_cumprod[t]) * mean(x0)
#     """
#     sum_x0 = 0.0
#     count = 0
#     for batch in digit_loader:
#         images, labels = batch
#         # Since this loader only contains digit 1, all images are used.
#         imgs = images.to(device)
#         sum_x0 += imgs.sum(dim=0)
#         count += imgs.shape[0]
#     global_mean = sum_x0 / count  # shape: (1, 28, 28)
    
#     target_means = {}
#     for t in checkpoints:
#         # model.sqrt_alphas_cumprod is registered as a buffer of shape [timesteps]
#         factor = model.sqrt_alphas_cumprod[t]
#         # Multiply global_mean (shape: [1,28,28]) by factor to get target mean at timestep t.
#         # We add a batch dimension so the target has shape (1, 1, 28, 28)
#         target_means[t] = factor * global_mean.unsqueeze(0)
#     return target_means

# #######################################
# # NEW: Reverse diffusion with noise sample search
# #######################################
# def reverse_diffusion_search(model, target_means, n_candidates=16, checkpoint_interval=100, device="cuda"):
#     """
#     Runs the reverse diffusion process starting from n_candidates noise samples.
#     At every checkpoint (every checkpoint_interval steps), scores the candidates against the target distribution
#     (using negative MSE loss) and keeps only the top half.
#     Returns the best candidate at the end.
#     """
#     # Initialize n_candidates noise samples (starting at timestep T)
#     candidates = torch.randn((n_candidates, model.in_channels, model.image_size, model.image_size), device=device)
    
#     for t in range(model.timesteps - 1, -1, -1):
#         # Create a tensor for the current timestep for each candidate
#         t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
#         # Sample random noise for the reverse diffusion update
#         noise = torch.randn_like(candidates)
#         # Update candidates using the reverse diffusion step (you can also try _reverse_diffusion_with_clip)
#         candidates = model._reverse_diffusion(candidates, t_tensor, noise)
        
#         # Every checkpoint_interval steps, score and prune candidates
#         if t % checkpoint_interval == 0:
#             if t in target_means:
#                 # Expand the target mean to match candidates shape
#                 target = target_means[t].expand(candidates.shape[0], -1, -1, -1)
#                 # Compute score: negative MSE (so lower error gives higher score)
#                 scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
#                 # Keep the top half candidates (at least 1 candidate)
#                 k = max(1, candidates.shape[0] // 2)
#                 topk_indices = torch.topk(scores, k=k).indices
#                 candidates = candidates[topk_indices]
#                 print(f"At timestep {t}, kept {candidates.shape[0]} candidates after scoring.")
    
#     # At t=0, optionally choose the best candidate
#     if 0 in target_means:
#         target = target_means[0].expand(candidates.shape[0], -1, -1, -1)
#         final_scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
#         best_idx = torch.argmax(final_scores)
#         best_candidate = candidates[best_idx]
#     else:
#         best_candidate = candidates[0]
#     return best_candidate

# #######################################
# # Visualization helper
# #######################################
# def plot_image(image_tensor, title="Generated Image"):
#     # The images are in [-1, 1]. Convert to [0, 1] for display.
#     image = (image_tensor.squeeze().detach().cpu().numpy() + 1) / 2.0
#     plt.figure(figsize=(3, 3))
#     plt.imshow(image, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# #######################################
# # Main function
# #######################################
# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     ckpt_path = os.path.join(TRAINED_MODELS_DIR, "epoch_100_steps_00046900.pt")
#     model = load_model(ckpt_path, device)
    
#     # Create a dataloader for digit 1 examples (for target distribution estimation)
#     digit_loader = create_digit_dataloader(digit=1, batch_size=128)
    
#     # Define checkpoints (every 100 timesteps; ensure these values are within [0, timesteps-1])
#     checkpoints = list(range(0, model.timesteps, 200))
#     print(f"Using checkpoints: {checkpoints}")
    
#     # Estimate the target forward diffusion distribution (here, the mean) for digit 1
#     target_means = estimate_target_distribution(model, digit_loader, checkpoints, device)
    
#     # Run reverse diffusion search: start with several candidate noises and prune at checkpoints
#     best_noise = reverse_diffusion_search(model, target_means, n_candidates=16, checkpoint_interval=200, device=device)
    
#     # To generate the final sample, run the reverse diffusion from best_noise fully if not already at t=0.
#     # (In our loop, we already ran until t=0.)
#     # Denormalize the final output from [-1,1] to [0,1] for visualization.
#     plot_image(best_noise, title="Reverse Diffusion Search Output for Digit 1")

# if __name__ == "__main__":
#     main()


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Directories and model import
MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'gpu_accelerated_training')
TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'trained_ddpm', 'results')
sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
sys.path.append(TRAINED_MODELS_DIR)
from model import MNISTDiffusion  # Adjust import path if needed

# Create MNIST dataloader for all digits (used for training)...
def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Create a dataloader that only returns images of a specified digit.
def create_digit_dataloader(digit, batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    full_dataset = datasets.MNIST(root="./mnist_data", train=True, download=True, transform=preprocess)
    # Filter indices for the specified digit.
    indices = [i for i, target in enumerate(full_dataset.targets) if target == digit]
    subset = Subset(full_dataset, indices)
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

#######################################
# Estimate target forward diffusion distribution for digit 1
#######################################
def estimate_target_distribution(model, digit_loader, checkpoints, device="cuda"):
    """
    Computes the target distribution q(x_t|y=1) for a set of checkpoint timesteps.
    For the closed-form forward diffusion, the mean is:
        x_t = sqrt(alphas_cumprod[t]) * x0.
    We compute the average x0 for digit 1 and then, for each checkpoint t,
    define target_mean[t] = sqrt(alphas_cumprod[t]) * mean(x0).
    """
    sum_x0 = 0.0
    count = 0
    for batch in digit_loader:
        images, labels = batch
        imgs = images.to(device)
        sum_x0 += imgs.sum(dim=0)
        count += imgs.shape[0]
    global_mean = sum_x0 / count  # shape: (1, 28, 28)
    
    target_means = {}
    for t in checkpoints:
        factor = model.sqrt_alphas_cumprod[t]
        # Ensure target_mean has shape (1, 1, 28, 28)
        target_means[t] = factor * global_mean.unsqueeze(0)
    return target_means

#######################################
# Reverse diffusion with noise sample search
#######################################
def reverse_diffusion_search(model, target_means, n_candidates=16, checkpoint_interval=100, device="cuda"):
    """
    Runs the reverse diffusion process starting from n_candidates noise samples.
    At every checkpoint (every checkpoint_interval steps), scores the candidates against
    the target distribution using negative MSE loss and retains the top half.
    Returns the best candidate at t=0.
    """
    candidates = torch.randn((n_candidates, model.in_channels, model.image_size, model.image_size), device=device)
    
    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)
        
        # Score and prune candidates at checkpoints
        if t % checkpoint_interval == 0:
            if t in target_means:
                target = target_means[t].expand(candidates.shape[0], -1, -1, -1)
                scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
                k = max(1, candidates.shape[0] // 2)
                topk_indices = torch.topk(scores, k=k).indices
                candidates = candidates[topk_indices]
                print(f"At timestep {t}, retained {candidates.shape[0]} candidate(s).")
    
    # At t=0, choose the best candidate.
    if 0 in target_means:
        target = target_means[0].expand(candidates.shape[0], -1, -1, -1)
        final_scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
        best_idx = torch.argmax(final_scores)
        best_candidate = candidates[best_idx]
    else:
        best_candidate = candidates[0]
    return best_candidate

#######################################
# Visualization helper
#######################################
def plot_image(image_tensor, title="Generated Image"):
    image = (image_tensor.squeeze().detach().cpu().numpy() + 1) / 2.0
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

#######################################
# Main function: Run the experiment multiple times
#######################################
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, "epoch_100_steps_00046900.pt")
    model = load_model(ckpt_path, device)
    
    # Create a dataloader for digit 1 examples for target distribution estimation.
    digit_loader = create_digit_dataloader(digit=1, batch_size=128)
    
    # Define checkpoints (every 100 timesteps)
    checkpoints = list(range(0, model.timesteps, 200))
    print(f"Using checkpoints: {checkpoints}")
    
    # Estimate the target distribution (mean) for digit 1.
    target_means = estimate_target_distribution(model, digit_loader, checkpoints, device)
    
    # Run the reverse diffusion search experiment 10 times.
    n_experiments = 10
    outputs = []
    for i in range(n_experiments):
        best_noise = reverse_diffusion_search(model, target_means, n_candidates=16, checkpoint_interval=200, device=device)
        outputs.append(best_noise)
        print(f"Experiment {i+1} complete.")
    
    # Plot the 10 outputs in a grid.
    num_cols = 5
    num_rows = (n_experiments + num_cols - 1) // num_cols
    plt.figure(figsize=(num_cols * 3, num_rows * 3))
    for idx, out in enumerate(outputs):
        plt.subplot(num_rows, num_cols, idx+1)
        plot_image(out, title=f"Run {idx+1}")
    plt.suptitle("Reverse Diffusion Search Outputs for Digit 1")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

