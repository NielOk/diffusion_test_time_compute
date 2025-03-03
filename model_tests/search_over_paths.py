import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model_visual2 import load_code, get_model_paths, load_model_architecture, get_denoising_steps

MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)

def scaling_experiment():
    min_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = 'nlc'  # 'lc' for label-conditioned, 'nlc' for non-label-conditioned
    results_filename = f'{model_type}_results.json'

    TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage = load_code(model_type=model_type)

    # Load data
    train_loader, test_loader = create_mnist_dataloaders(batch_size=128,image_size=28)

    # Get model filepaths
    sorted_model_paths, sorted_epoch_numbers = get_model_paths()

    for i in range(len(sorted_model_paths)):
        model_path = sorted_model_paths[i]
        epoch_number = sorted_epoch_numbers[i]

        # Load model architecture
        model = load_model_architecture(MNISTDiffusion, device=device, model_type=model_type)
        model_ema = ExponentialMovingAverage(model, decay=0.995, device=device)

        # Load model weights
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])

def singular_model_experiment():
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
    #scaling_experiment()
    singular_model_experiment()