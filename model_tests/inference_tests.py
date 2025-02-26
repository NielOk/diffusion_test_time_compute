import os
import sys
import torch
import torch.nn as nn

MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'gpu_accelerated_training')
TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'trained_ddpm', 'results')

sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
from train_mnist import create_mnist_dataloaders
from model import MNISTDiffusion
from utils import ExponentialMovingAverage

from test_models import get_model_paths

# Function to visualize model inference
def visualize_inference(
        model_path, # Model to load
        device="cpu", # Device to run model o
        timesteps=1000, # Number of steps to run model for
        image_size=28, # Size of image
        base_dim=64, # Base dimension of Unet
    ):

    # Load model
    model = MNISTDiffusion(timesteps=timesteps,
                image_size=image_size,
                in_channels=1,
                base_dim=base_dim,
                device=device)
    
    model_ema = ExponentialMovingAverage(model, decay=0.995)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model.load_state_dict(checkpoint['model_ema'])