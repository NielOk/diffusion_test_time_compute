''' 
Metrics-based testing of models trained on MNIST dataset using DDPM.
'''

import os
import sys
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'gpu_accelerated_training')
LC_TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'lc_trained_ddpm', 'results') # lc means label-conditioned, nlc means non-label-conditioned

sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
from train_mnist import create_mnist_dataloaders
from model import MNISTDiffusion
from utils import ExponentialMovingAverage

def get_model_paths():
    # Get model filepaths
    epoch_numbers = []
    model_paths = []
    for filename in os.listdir(LC_TRAINED_MODELS_DIR):
        if filename.endswith('.pt'):
            epoch_number = int(filename.split('epoch_')[1].split('_steps_')[0])
            epoch_numbers.append(epoch_number)
            model_paths.append(os.path.join(LC_TRAINED_MODELS_DIR, filename))
    
    # Sort model filepaths by epoch number
    sorted_model_paths = [f for _, f in sorted(zip(epoch_numbers, model_paths))]
    sorted_epoch_numbers = sorted(epoch_numbers)

    return sorted_model_paths, sorted_epoch_numbers

# Test models just based on the loss and ema loss
def test_models_loss(
        min_epoch=1, # Minimum epoch number to start testing from
        device="cpu", # Device to run model on
        results_filename="results.json" # File to save results to
        ):
    # Load data
    train_loader, test_loader = create_mnist_dataloaders(batch_size=128,image_size=28)

    # Get model filepaths
    sorted_model_paths, sorted_epoch_numbers = get_model_paths()

    model_evals = {}
    for i in range(len(sorted_model_paths)):
        model_path = sorted_model_paths[i]
        epoch_number = sorted_epoch_numbers[i]
        print(f'Testing model at epoch {epoch_number}')

        model_evals[epoch_number] = {'model_path': model_path, 
                                     'model_losses': [],
                                     'model_ema_losses': []}

        if epoch_number < min_epoch: 
            continue

        model=MNISTDiffusion(timesteps=1000,
                image_size=28,
                in_channels=1,
                num_classes=10,
                base_dim=64,
                dim_mults=[2,4]).to(device)
        
        model_ema=ExponentialMovingAverage(model, decay=0.995, device=device)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])

        # Test model
        model.eval()
        loss_fn = nn.MSELoss(reduction='mean')
        for i, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.to(device)
            noise = torch.randn_like(image).to(device)
            pred_noise = model(image, noise, target)
            loss = loss_fn(pred_noise, noise)
            model_evals[epoch_number]['model_losses'].append(loss.item())

            ema_pred_noise = model_ema(image, noise, target)
            ema_loss = loss_fn(ema_pred_noise, noise)
            model_evals[epoch_number]['model_ema_losses'].append(ema_loss.item())

        # Save results
        results_filepath = os.path.join(REPO_DIR, results_filename)
        with open(results_filepath, 'w') as f:
            json.dump(model_evals, f)

# Analyze eval data for loss 
def analyze_eval_data_loss(
        min_epoch=1, # Minimum epoch number to start analyzing from
        results_filename='results.json'
        ):
    
    # Load results
    results_filepath = os.path.join(MODEL_TEST_DIR, 'test_results', results_filename)
    with open(results_filepath, 'r') as f:
        results = json.load(f)

    mean_model_losses = []
    mean_model_ema_losses = []
    epoch_numbers = []
    # Analyze results
    for epoch_number, model_eval in results.items():
        epoch_numbers.append(int(epoch_number))

        model_losses = model_eval['model_losses']
        model_ema_losses = model_eval['model_ema_losses']

        average_model_loss = sum(model_losses) / len(model_losses)
        average_model_ema_loss = sum(model_ema_losses) / len(model_ema_losses)
        
        mean_model_losses.append(average_model_loss)
        mean_model_ema_losses.append(average_model_ema_loss)

    # Graph results
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_numbers, mean_model_losses, label='Model Loss')
    plt.plot(epoch_numbers, mean_model_ema_losses, label='Model EMA Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    # Define custom tick positions (1, 5, 10, 15, 20, ...)
    tick_positions = list(range(5, max(epoch_numbers) + 1, 5))
    plt.xticks(tick_positions, rotation=90) 
    plt.legend()
    plt.show()

if __name__ == '__main__':
    min_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_filename = 'results.json'
    
    # Collect eval data
    #test_models_loss(min_epoch=min_epoch, device=device, results_filename=results_filename) # Comment out once data has been collected

    # Analyze eval data
    analyze_eval_data_loss(min_epoch=min_epoch, results_filename=results_filename)