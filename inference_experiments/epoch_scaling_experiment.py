'''
An experiment to test the effect of scaling the number of epochs on the performance of the model with various 
inference methods.
'''

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import datetime

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Transformers / HF
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

from inference_experiment_utils import * 

# =============
# Constants
# =============

MODEL_TYPE = "lc" # "lc" or "nlc" for label-conditioned or non-label-conditioned models
USE_EMA = True # Use EMA weights for inference
USE_CLIP = True # Use clipping for inference
INFERENCE_APPROACHES = ["reverse-diffusion", "search-over-paths"] # Possible inference approaches
SCORING_APPROACHES = ["mse", "bayes", "mixture", "search-over-paths"] # Possible scoring approaches
N_EXPERIMENTS_PER_DIGIT = 20 # Number of times to run the experiment for each digit
VERIFIER_DATA_SIZE = 400 # Number of samples to use for the verifier. Should be chosen after experiments with the verifier scaling first.

# Number of candidates for search-over-paths
N_CANDIDATES = 16

# Hugging Face MNIST classifier repository
HF_MODEL_NAME = "farleyknight/mnist-digit-classification-2022-09-04"

# Data
MNIST_ROOT = "./mnist_data"
DEBUG_VISUALIZE_DIGIT = False  # If True, will show a sample digit from the subset

# Directories and model import
INFERENCE_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(INFERENCE_EXPERIMENTS_DIR)

TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage = load_code(model_type=MODEL_TYPE)

DIGIT_ARRAY = list(range(10))

sys.path.append(TRAINED_MODELS_DIR)

# =============
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join(INFERENCE_EXPERIMENTS_DIR, f"experiment_outputs_{timestamp}")
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"All results, data, & figures will be saved to: {RESULTS_DIR}")

# ============================
# Global Helper / Cache
# ============================
_distribution_cache = {}  # key: (digit, subset_size, approach) -> precomputed_data

def singular_experiment():
    ### Parameters picked by user ###
    n_candidates = 3
    digit_to_generate = 8
    delta_f = 30 # Number of steps to renoise back to
    delta_b = 60 # Number of steps to denoise back to

    # Search over paths
    ema = True # Use EMA for this test
    n_samples = 1
    use_clip = True
    labels = None
    scoring_approach = "mse"

    ### More generalized code ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = 'nlc'  # 'lc' for label-conditioned, 'nlc' for non-label-conditioned

    # Load code
    TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage = load_code(model_type=model_type)

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

    model_ema.eval()
    model.eval()

    subset_size = 1000

    digit_loader = create_digit_dataloader(digit=digit_to_generate, subset_size=subset_size, batch_size=128)

    # Get mse target distribution
    inference_checkpoints = get_checkpoints(num_steps=1000, delta_f=delta_f, delta_b=delta_b)
    target_distribution = estimate_target_distribution_mse(model, digit_loader, digit_to_generate, inference_checkpoints, device=device)

    # Search over paths
    best_candidate = search_over_paths(n_candidates, delta_f, delta_b, model, model_ema, digit_to_generate, target_distribution, model_type=model_type, ema=ema, use_clip=use_clip, device=device, scoring_approach=scoring_approach)

    # Draw best candidate
    best_image = (best_candidate.squeeze().detach().cpu().numpy() + 1.0) / 2.0
    plt.imshow(best_image, cmap='gray')
    plt.title("Best Generated Image")
    plt.show()

if __name__ == '__main__':
    singular_experiment()