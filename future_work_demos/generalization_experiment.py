import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # If running on a headless server
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import datetime
import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
from collections import defaultdict

# Transformers / HF
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

FUTURE_WORK_DEMOS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(FUTURE_WORK_DEMOS_DIR)
INFERENCE_EXPERIMENTS_DIR = os.path.join(REPO_DIR, "inference_experiments")
HEMAL_EXPERIMENTS_DIR = os.path.join(REPO_DIR, "hemal_experiments")
MNIST_ROOT_FW = "./mnist_data"
sys.path.append(INFERENCE_EXPERIMENTS_DIR)

from inference_experiment_utils import *
from fid_classifier import Classifier

def normalized_generate_images_and_get_verifier_images(model, model_ema, digit_to_generate, n_samples, verifier_data_size, n_candidates=5, delta_f=10, delta_b=30, approach="mse", search_method="paths", model_type="nlc", device='cpu', ema=True, use_clip=True):

    generated_samples = []

    # Generated samples are already normalized to [-1, 1]
    digit_samples, verifier_data_indices = generate_samples_for_digit(
        model=model, 
        model_ema=model_ema,
        digit_to_generate=digit_to_generate,
        verifier_data_subset_size = verifier_data_size, 
        n_candidates=n_candidates,
        delta_f=delta_f, 
        delta_b=delta_b,
        model_type=model_type,
        approach=approach,
        search_method=search_method,
        n_experiments=n_samples, 
        device=device,
        ema=ema,
        use_clip=use_clip
    )

    verifier_images, verifier_labels = load_data_from_indices(verifier_data_indices, root=MNIST_ROOT_FW, train=True, download=True)

    # FID classifier is trained on MNIST normalized to [-1, 1]
    to_tensor = transforms.ToTensor()
    normalized_verifier_images = [2.0 * (to_tensor(img) - 0.5) for img in verifier_images]

    # Check if the verifier labels match the digit to generate in case
    for verifier_label in verifier_labels:
        if verifier_label != digit_to_generate:
            raise ValueError(f"Verifier label {verifier_label} does not match digit to generate {digit_to_generate}")

    return digit_samples, normalized_verifier_images

def main():

    # Main settings
    subset_size = 400
    n_samples = 1 #50

    # Configurations
    digit_array = list(range(10))
    delta_f = 100
    delta_b = 200
    n_candidates_paths = 2 #5
    ema=True
    use_clip=True
    scoring_approach = "mse"
    search_method = "paths"

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

    model.eval()
    model_ema.eval()

    # Get the normalized generated samples and verifier samples
    digit_samples, verifier_samples = normalized_generate_images_and_get_verifier_images(
        model=model,
        model_ema=model_ema,
        digit_to_generate=8,
        n_samples=n_samples,
        verifier_data_size=subset_size,
        n_candidates=n_candidates_paths,
        delta_f=delta_f,
        delta_b=delta_b,
        approach=scoring_approach,
        search_method=search_method,
        model_type=model_type,
        device=device,
        ema=ema,
        use_clip=use_clip
    )

    # Create the generalization experiment results directory first
    generalization_experiment_results_dir = os.path.join(FUTURE_WORK_DEMOS_DIR, "generalization_experiment_results")
    os.makedirs(generalization_experiment_results_dir, exist_ok=True)

    # Save the generated samples
    save_dir = os.path.join(generalization_experiment_results_dir, "generated_samples")
    os.makedirs(save_dir, exist_ok=True)
    for i, (img, label) in enumerate(digit_samples):
        img = img.squeeze().detach().cpu().numpy()
        img = (img + 1.0) / 2.0 * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img, mode='L').convert('RGB')
        img_pil.save(os.path.join(save_dir, f"generated_sample_{i}.png"))

    # Save 2 verifier samples
    save_dir = os.path.join(generalization_experiment_results_dir, "verifier_samples")
    os.makedirs(save_dir, exist_ok=True)
    for i, sample in enumerate(verifier_samples):
        if i >= 2:
            break
        img = Image.fromarray((sample[0].cpu().numpy() * 255).astype(np.uint8), mode='L')
        img.save(os.path.join(save_dir, f"verifier_sample_{i}.png"))

if __name__ == '__main__':
    main()