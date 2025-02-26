import os
import sys
import torch

MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'gpu_accelerated_training')
TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'trained_ddpm', 'results')

sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
from train_mnist import create_mnist_dataloaders
from model import MNISTDiffusion
from utils import ExponentialMovingAverage

def test_all_models_cpu():
    # Load data
    train_loader, test_loader = create_mnist_dataloaders(batch_size=128,image_size=28)

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

    for model_path in sorted_model_paths:
        device="cpu"
        model=MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                base_dim=64,
                dim_mults=[2,4]).to(device)
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model'])
    
    

if __name__ == '__main__':
    test_all_models_cpu()