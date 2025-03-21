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
MNIST_ROOT = "./mnist_data"
sys.path.append(INFERENCE_EXPERIMENTS_DIR)

from inference_experiment_utils import *
from fid_classifier import Classifier

def generate_and_save_images(model, digit_to_generate, n_samples, verifier_data_size, approach="mse", search_method="paths", device='cpu'):

    generated_samples = []



if __name__ == '__main__':
    digit_loader, verifier_data_indices = create_digit_dataloader(digit=8, subset_size = 100, batch_size=128)
    for batch in digit_loader:
        images, labels = batch

        # Convert image to numpy, denormalize if necessary
        img = images[99].squeeze().cpu().numpy()
        img = (img * 0.5) + 0.5  # undo Normalize([-1, 1] scaling)

        # Plot and save
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.savefig("digit8_sample.png", bbox_inches='tight')
        break  # only save one image

    full_dataset = datasets.MNIST(root=MNIST_ROOT, train=True, download=True)
    images, labels = zip(*[full_dataset[i] for i in verifier_data_indices])
    img = images[99]
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig("digit8_sample_full.png", bbox_inches='tight')
    print("Saved sample images.")