import os
import sys
import json
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

def test_models(
        min_epoch = 1, # Minimum epoch number to start testing from
        device = "cpu", # Device to run model on
        results_filename = "results.json" # File to save results to
        ):
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
    sorted_epoch_numbers = sorted(epoch_numbers)

    model_evals = {}
    for i in range(len(sorted_model_paths)):
        model_path = sorted_model_paths[i]
        epoch_number = sorted_epoch_numbers[i]
        print(f'Testing model at epoch {epoch_number}')

        model_evals[epoch_number] = {'model_path': model_path, 
                                     'losses': []}

        if epoch_number < min_epoch: 
            continue

        model=MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                base_dim=64,
                dim_mults=[2,4]).to(device)
        if device == "cpu":
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint['model'])
        elif device == "cuda":
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model'])

        # Test model
        model.eval()
        loss_fn = nn.MSELoss(reduction='mean')
        for i, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device)
            with torch.no_grad():
                noise = torch.randn_like(image)
                pred_noise = model(image, noise)
                loss = loss_fn(pred_noise, noise)
                model_evals[epoch_number]['losses'].append(loss.item())

        # Save results
        with open(results_filename, 'w') as f:
            json.dump(model_evals, f)

def analyze_eval_data(
        results_filename='results.json'
        ):
    pass
    
if __name__ == '__main__':
    min_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_filename = 'results.json'
    
    # Collect eval data
    test_models(min_epoch=min_epoch, device=device)

    # Analyze eval data