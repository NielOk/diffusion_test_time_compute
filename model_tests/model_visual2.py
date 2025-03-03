import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import inspect

MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)

def load_code(model_type='lc'):
    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')
    
    if 'model' in sys.modules:
        del sys.modules['model']
    if 'utils' in sys.modules:
        del sys.modules['utils']
    if 'train_mnist' in sys.modules:
        del sys.modules['train_mnist']
    if 'unet' in sys.modules:
        del sys.modules['unet']

    # Get directories
    lc_gpu_accelerated_training = os.path.join(REPO_DIR, 'lc_gpu_accelerated_training')
    LC_TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'lc_trained_ddpm', 'results')
    nlc_gpu_accelerated_training = os.path.join(REPO_DIR, 'nlc_gpu_accelerated_training')
    NLC_TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'nlc_trained_ddpm', 'results')

    if model_type == 'lc':
        sys.path.append(lc_gpu_accelerated_training)

        if nlc_gpu_accelerated_training in sys.path:
            sys.path.remove(nlc_gpu_accelerated_training)
        
        from train_mnist import create_mnist_dataloaders
        from model import MNISTDiffusion
        from utils import ExponentialMovingAverage

        return LC_TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage

    elif model_type == 'nlc':
        sys.path.append(nlc_gpu_accelerated_training)
        if lc_gpu_accelerated_training in sys.path:
            sys.path.remove(lc_gpu_accelerated_training)

        from train_mnist import create_mnist_dataloaders
        from model import MNISTDiffusion
        from utils import ExponentialMovingAverage

        return NLC_TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage
    
def get_model_paths(TRAINED_MODELS_DIR):
    # Get model filepaths
    epoch_numbers = []
    model_paths = []
    for filename in os.listdir(TRAINED_MODELS_DIR):
        if filename.endswith('.pt'):
            epoch_number = int(filename.split('epoch_')[1].split('_steps_')[0])
            epoch_numbers.append(epoch_number)
            model_paths.append(os.path.join(TRAINED_MODELS_DIR, filename))
    
    # Sort model filepaths by epoch number
    sorted_model_paths = [f for _, f in sorted(zip(epoch_numbers, model_paths))]
    sorted_epoch_numbers = sorted(epoch_numbers)

    return sorted_model_paths, sorted_epoch_numbers

def load_model_architecture(MNISTDiffusion,device='cpu', model_type='lc'):
    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')
    elif model_type == 'lc':
        model = MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                num_classes=10,
                base_dim=64,
                dim_mults=[2,4]).to(device)
    else:
        model = MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                base_dim=64,
                dim_mults=[2,4]).to(device)
    
    return model

def get_denoising_steps(model,
                    n_samples,
                    model_type='lc', 
                    device='cpu',
                    steps_to_show=20, 
                    use_clip=True, # whether to use _reverse_diffusion_with_clip or _reverse_diffusion. Not to do with OpenAI clip. 
                    labels=None # only used if model_type='lc'. Should be a list of desired labels for each sample, which will be converted to a tensor of shape (n_samples,)
                    ):
    
    if len(labels) != n_samples:
        raise ValueError('labels must be a list of length n_samples')
    
    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')
    
    # Convert labels to tensor
    labels = torch.tensor(labels, device=device)
    
    timesteps = model.timesteps

    # Generate random noise
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

        if model_type == 'lc':
            if use_clip:
                x_t = model._reverse_diffusion_with_clip(x_t, t, noise, labels)
            else:
                x_t = model._reverse_diffusion(x_t, t, noise, labels)
        elif model_type == 'nlc':
            if use_clip:
                x_t = model._reverse_diffusion_with_clip(x_t, t, noise)
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

def visualizer_single_model_experiment():
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

    # Search over paths
    n_samples = 4
    steps_to_show = 20
    use_clip = True
    labels = [0, 1, 2, 3]
    images_over_time = get_denoising_steps(model, n_samples, model_type=model_type, device=device, steps_to_show=steps_to_show, use_clip=use_clip, labels=labels)

    # Plot the results
    plot_denoising_process(images_over_time, n_samples=n_samples)

if __name__ == '__main__':
    visualizer_single_model_experiment()