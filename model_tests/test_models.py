''' 
Metrics-based testing of models trained on MNIST dataset using DDPM. These assume the models are lc_trained_ddpm models.
'''

import os
import sys
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)

def load_code(model_type='lc'):
    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')
    elif model_type == 'lc':
        GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'lc_gpu_accelerated_training')
        TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'lc_trained_ddpm', 'results')
        sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
    else:
        GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'nlc_gpu_accelerated_training')
        TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'nlc_trained_ddpm', 'results')
        sys.path.append(GPU_ACCELERATED_TRAINING_DIR)

    from train_mnist import create_mnist_dataloaders
    from model import MNISTDiffusion
    from utils import ExponentialMovingAverage
    
    return TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage

def get_model_paths():
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

def load_model_architecture(device='cpu', model_type='lc'):
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

# Test models just based on the loss and ema loss
def test_models_loss(
        min_epoch=1, # Minimum epoch number to start testing from
        device="cpu", # Device to run model on
        model_type='lc', # 'lc' for label-conditioned, 'nlc' for non-label-condition
        results_filename="results.json" # File to save results to
        ):
    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')

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

        # Load model architecture
        model = load_model_architecture(device=device, model_type='nlc')
        
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
            if model_type == 'lc':
                pred_noise = model(image, noise, target)
            else:
                pred_noise = model(image, noise)
            loss = loss_fn(pred_noise, noise)
            model_evals[epoch_number]['model_losses'].append(loss.item())

            if model_type == 'lc':
                ema_pred_noise = model_ema(image, noise, target)
            else:
                ema_pred_noise = model_ema(image, noise)

            ema_loss = loss_fn(ema_pred_noise, noise)
            model_evals[epoch_number]['model_ema_losses'].append(ema_loss.item())

        # Save results
        results_filepath = os.path.join(REPO_DIR, results_filename)
        with open(results_filepath, 'w') as f:
            json.dump(model_evals, f)

# Analyze eval data for loss 
def analyze_eval_data_loss(
        min_epoch=1, # Minimum epoch number to start analyzing from
        results_filename='results.json',
        model_type='lc' # 'lc' for label-conditioned, 'nlc' for non-label-conditioned
        ):
    
    # Load results
    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')
    
    if model_type == 'lc':
        results_filepath = os.path.join(MODEL_TEST_DIR, 'lc_test_results', results_filename)
    else:
        results_filepath = os.path.join(MODEL_TEST_DIR, 'nlc_test_results', results_filename)
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
    plt.plot(epoch_numbers, mean_model_losses, label=f'Model Loss ({model_type})')
    plt.plot(epoch_numbers, mean_model_ema_losses, label=f'Model EMA Loss ({model_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    # Define custom tick positions (1, 5, 10, 15, 20, ...)
    tick_positions = list(range(5, max(epoch_numbers) + 1, 5))
    plt.xticks(tick_positions, rotation=90) 
    plt.title(f'Loss vs. Epoch ({model_type})')
    plt.legend()
    plt.savefig(os.path.join(MODEL_TEST_DIR, f'loss_vs_epoch_{model_type}.png'))

if __name__ == '__main__':
    min_epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = 'nlc'  # 'lc' for label-conditioned, 'nlc' for non-label-conditioned
    results_filename = f'{model_type}_results.json'

    TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage = load_code(model_type=model_type)
    
    # Collect eval data
    #test_models_loss(min_epoch=min_epoch, device=device, model_type=model_type, results_filename=results_filename) # Comment out once data has been collected

    # Analyze eval data
    analyze_eval_data_loss(min_epoch=min_epoch, results_filename=results_filename, model_type=model_type) # Comment out when collecting data on gpu