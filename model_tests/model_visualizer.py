# visualize_denoising.py

import torch
import matplotlib.pyplot as plt
import math
import os
import sys

MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
# GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'gpu_accelerated_training')

LC = False
LABEL = torch.tensor([6])

if LC:
    GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'lc_gpu_accelerated_training')
    TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'lc_trained_ddpm', 'results')

else:
    GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'nlc_gpu_accelerated_training')
    TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'nlc_trained_ddpm', 'results')

sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
sys.path.append(TRAINED_MODELS_DIR)

from model import MNISTDiffusion  # Adjust if your import path is different

CHECKPOINT = "epoch_100_steps_00046900.pt"

def load_model(ckpt_path, lc=True, device="cuda"):
    """
    Loads the MNISTDiffusion model with weights from the given checkpoint path.
    """
    # Create an instance of the model (must match training hyperparams)
    model = MNISTDiffusion(
        timesteps=1000,   # Must match the timesteps used in training
        image_size=28,    # MNIST image size
        in_channels=1,    # MNIST is 1-channel
        base_dim=64,      # Must match --model_base_dim
        dim_mults=[2, 4]  # Must match the model config used in training
    )
    model.to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # If you want the EMA weights for best results, load from "model_ema"
    if "model_ema" in ckpt:
        ema_state = ckpt["model_ema"]
        # The averaged parameters are stored under "module" in EMA's state_dict
        if "module" in ema_state:
            model.load_state_dict(ema_state["module"], strict=False)
        else:
            # Fallback if your checkpoint uses a different key
            model.load_state_dict(ckpt["model"], strict=False)
    else:
        # Otherwise load the raw model weights
        model.load_state_dict(ckpt["model"], strict=False)

    model.eval()
    return model


@torch.no_grad()
def get_denoising_steps(model, n_samples=1, device="cuda", steps_to_show=10, use_clip=True):
    """
    Performs reverse diffusion step-by-step from random noise, 
    returns a list of images at selected timesteps so we can
    visualize how the noise is denoised over time.

    Args:
        model: MNISTDiffusion instance (already loaded and in eval mode).
        n_samples: how many samples (digits) to generate at once.
        device: "cuda" or "cpu".
        steps_to_show: number of intermediate steps to collect for visualization.
        use_clip: whether to use the clipped reverse diffusion.

    Returns:
        A list of (timestep, images) pairs, where images have shape
        [n_samples, 1, 28, 28].
    """
    timesteps = model.timesteps
    x_t = torch.randn(
        (n_samples, model.in_channels, model.image_size, model.image_size), 
        device=device
    )

    # Pick which timesteps to store for plotting
    steps_indices = torch.linspace(timesteps - 1, 0, steps_to_show, dtype=torch.int).tolist()
    steps_indices = sorted(list(set(int(s) for s in steps_indices)), reverse=True)

    images_over_time = []  # will store (timestep, x_t) pairs

    for i in range(timesteps - 1, -1, -1):
        t = torch.tensor([i] * n_samples, device=device)
        noise = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t)

        if use_clip:
            if LC:
                x_t = model._reverse_diffusion_with_clip(x_t, t, noise, labels=LABEL)
            else:
                x_t = model._reverse_diffusion_with_clip(x_t, t, noise)
        else:
            if LC:
                x_t = model._reverse_diffusion(x_t, t, noise, labels=LABEL)
            else:
                x_t = model._reverse_diffusion(x_t, t, noise)

        # Save intermediate steps if i is in our selected list
        if i in steps_indices:
            x_t_for_plot = (x_t + 1) / 2.0  # from [-1,1] to [0,1]
            images_over_time.append((i, x_t_for_plot.clamp(0,1).cpu()))

    # Ensure t=0 is included if not already
    if 0 not in steps_indices:
        x_t_for_plot = (x_t + 1) / 2.0
        images_over_time.append((0, x_t_for_plot.clamp(0,1).cpu()))

    # Sort by timestep descending
    images_over_time.sort(key=lambda x: x[0], reverse=True)
    return images_over_time


def plot_denoising_process(images_over_time, n_samples=1):
    """
    Given a list of (timestep, images) pairs, plots them in a single figure.
    images is shape [n_samples, 1, 28, 28].
    """
    n_plots = len(images_over_time)
    fig, axes = plt.subplots(n_samples, n_plots, figsize=(3*n_plots, 3*n_samples))

    # Handle the case of n_samples=1
    if n_samples == 1:
        axes = [axes]

    for row_sample in range(n_samples):
        for col_step in range(n_plots):
            ax = axes[row_sample][col_step] if n_samples > 1 else axes[col_step]
            t_step, imgs = images_over_time[col_step]
            # imgs: [n_samples, 1, 28, 28]
            img = imgs[row_sample].squeeze(0).detach().numpy()  # shape [28, 28]
            ax.imshow(img, cmap='gray')
            ax.set_title(f"t={t_step}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # <-- Replace with your actual checkpoint file
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, CHECKPOINT)

    # 1. Load the model
    model = load_model(ckpt_path, device=device)

    # 2. Run step-by-step reverse diffusion
    n_samples = 4           # how many digits to generate in parallel
    steps_to_show = 8       # how many intermediate frames to visualize
    use_clip = True         # whether to use _reverse_diffusion_with_clip

    images_over_time = get_denoising_steps(
        model,
        n_samples=n_samples,
        device=device,
        steps_to_show=steps_to_show,
        use_clip=use_clip
    )

    # 3. Plot the results
    plot_denoising_process(images_over_time, n_samples=n_samples)


if __name__ == "__main__":
    main()
