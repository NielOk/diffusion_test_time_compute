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
import random
import scipy

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
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

def normalized_generate_images_and_get_verifier_indices(model, model_ema, digit_to_generate, n_samples, verifier_data_size, n_candidates=5, delta_f=10, delta_b=30, approach="mse", search_method="paths", model_type="nlc", device='cpu', ema=True, use_clip=True):

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

    return digit_samples, verifier_data_indices

def get_penultimate_features(model, x):
    """
    Get the penultimate features from the model.
    """
    with torch.no_grad():
        features = model.features(x)
    
    return features

class FlatImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, f)
                            for f in os.listdir(folder_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # Ensure grayscale
        if self.transform:
            img = self.transform(img)
        return img
    
def get_non_verifier_indices_dict(verifier_indices_dict):
    """
    Get non-verifier indices from the verifier indices.
    """

    all_verifier_indices = []
    for digit, verifier_indices in verifier_indices_dict.items():
        all_verifier_indices.extend(verifier_indices)

    all_indices = set(range(60000))
    all_verifier_indices_set = set(all_verifier_indices)
    non_verifier_indices = list(all_indices - all_verifier_indices_set)

    images, labels = load_data_from_indices(non_verifier_indices, root=MNIST_ROOT_FW, train=True, download=True)
    
    non_verifier_indices_dict = {}
    for digit in range(10):
        non_verifier_indices_dict[f"{digit}"] = []
    for i, label in enumerate(labels):
        non_verifier_indices_dict[f"{label}"].append(non_verifier_indices[i])

    return non_verifier_indices_dict

def get_features_for_folder(model, folder_path, batch_size=32, num_workers=4, device='cpu'):
    """
    Get penultimate features for all images in a folder (no subdirectories).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                  # (C, H, W), values in [0, 1]
        transforms.Normalize((0.5,), (0.5,))    # Normalize to [-1, 1]
    ])

    dataset = FlatImageFolderDataset(folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    features_list = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            features = get_penultimate_features(model, batch)  # You must define this
            features_list.append(features.cpu())

    return torch.cat(features_list, dim=0)

# Have to sample from the verifier data to get the features to have same number as the generated samples
def get_features_for_specific_indices(model, num_generated_samples, specific_indices, batch_size=32, num_workers=4, device='cpu'):
    """
    Get penultimate features for specific verifier indices.
    """
    
    specific_images, specific_labels = load_data_from_indices(specific_indices, root=MNIST_ROOT_FW, train=True, download=True)

    # Randomly sample num_generated_samples indices from the verifier set
    total = len(specific_images)
    sample_indices = random.sample(range(total), num_generated_samples)

    sampled_images = [specific_images[i] for i in sample_indices]
    sampled_labels = [specific_labels[i] for i in sample_indices]

    # Normalize and reshape
    to_tensor = transforms.ToTensor()
    sampled_normalized_verifier_images = torch.stack([
        2.0 * (transforms.ToTensor()(img) - 0.5) for img in sampled_images
    ])

    sampled_labels = torch.tensor(sampled_labels)

    # Create DataLoader
    specific_dataset = torch.utils.data.TensorDataset(sampled_normalized_verifier_images, sampled_labels)
    specific_loader = DataLoader(specific_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    features_list = []

    with torch.no_grad():
        for x_batch, _ in specific_loader:
            x_batch = x_batch.to(device)
            feats = model.features(x_batch)  # or use get_penultimate_features(model, x_batch)
            features_list.append(feats.cpu())

    return torch.cat(features_list, dim=0)

# Freched distance function
def frechet_distance(x_a, x_b):
    """
    Compute Fréchet distance between two sets of vectors x_a, x_b
    x_a shape => (N, d)
    x_b shape => (M, d)
    """
    mu_a    = np.mean(x_a, axis=0)
    sigma_a = np.cov(x_a.T)
    mu_b    = np.mean(x_b, axis=0)
    sigma_b = np.cov(x_b.T)

    diff = mu_a - mu_b
    # sqrtm can yield complex values if the covariance matrices are nearly singular.
    covmean, _ = scipy.linalg.sqrtm(sigma_a @ sigma_b, disp=False)
    # If there is a tiny imaginary component, just drop it.
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = np.sum(diff**2) + np.trace(sigma_a + sigma_b - 2.0*covmean)
    return fd

def compute_fid_score(fid_model_location, num_times_compute_fid, num_generated_samples, generated_results_dir, verifier_indices, non_verifier_indices, device='cpu'):
    
    model = Classifier()

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model weights
    model.load_state_dict(torch.load(fid_model_location, map_location=device))
    model.to(device)
    model.eval()

    # Get the features for the generated samples
    generated_features = get_features_for_folder(model, generated_results_dir, device=device)
    
    # Take samples num_times_compute_fid times
    total_against_verifier_data_fid = 0
    total_against_non_verifier_data_fid = 0
    all_measured_against_verifier_data_fids = []
    all_measured_against_non_verifier_data_fids = []
    for i in range(num_times_compute_fid):
        # Get the features for the verifier samples
        verifier_features = get_features_for_specific_indices(model, num_generated_samples, verifier_indices, device=device)

        # Get the features for the non-verifier samples
        non_verifier_features = get_features_for_specific_indices(model, num_generated_samples, non_verifier_indices, device=device)

        against_verifier_data_fid = frechet_distance(generated_features.cpu().numpy(), 
                            verifier_features.cpu().numpy())
        
        against_non_verifier_data_fid = frechet_distance(generated_features.cpu().numpy(),
                            non_verifier_features.cpu().numpy())
        
        total_against_verifier_data_fid += against_verifier_data_fid
        total_against_non_verifier_data_fid += against_non_verifier_data_fid
        all_measured_against_verifier_data_fids.append(against_verifier_data_fid)
        all_measured_against_non_verifier_data_fids.append(against_non_verifier_data_fid)

    # Compute the mean FID score
    mean_against_verifier_data_fid = total_against_verifier_data_fid / num_times_compute_fid
    mean_against_non_verifier_data_fid = total_against_non_verifier_data_fid / num_times_compute_fid

    return mean_against_verifier_data_fid, mean_against_non_verifier_data_fid, all_measured_against_verifier_data_fids, all_measured_against_non_verifier_data_fids

def main():

    # Main settings
    subset_size = 400
    n_samples = 50

    # Configurations
    digit_array = list(range(10))
    delta_f = 100
    delta_b = 200
    n_candidates_paths = 5
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

    # Create the generalization experiment results directory first
    generalization_experiment_results_dir = os.path.join(FUTURE_WORK_DEMOS_DIR, "generalization_experiment_results")
    os.makedirs(generalization_experiment_results_dir, exist_ok=True)


    ### Save generated samples and verifier indices ###
    verifier_indices_dict_path = os.path.join(generalization_experiment_results_dir, "verifier_indices.json")
    
    verifier_indices_dict = {}
    # Loop through digits
    for digit in digit_array:

        # Get the normalized generated samples and verifier samples
        digit_samples, verifier_indices = normalized_generate_images_and_get_verifier_indices(
            model=model,
            model_ema=model_ema,
            digit_to_generate=digit,
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
        
        verifier_indices_dict[digit] = verifier_indices

        # Save the generated samples
        save_dir = os.path.join(generalization_experiment_results_dir, f"digit_{digit}_generated_samples")
        os.makedirs(save_dir, exist_ok=True)
        for i, (img, label) in enumerate(digit_samples):
            img = img.squeeze().detach().cpu().numpy()
            img = (img + 1.0) / 2.0 * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img, mode='L').convert('RGB')
            img_pil.save(os.path.join(save_dir, f"generated_sample_{i}.png"))

    with open(verifier_indices_dict_path, 'w') as f:
        json.dump(verifier_indices_dict, f)

    ### Compute FID Score ###
    # Get FID model location
    fid_model_location = os.path.join(HEMAL_EXPERIMENTS_DIR, "classifier_20epochs.pt")

    # Load verifier indices
    with open(verifier_indices_dict_path, 'r') as f:
        verifier_indices_dict = json.load(f)

    # Get non-verifier indices
    non_verifier_indices_dict = get_non_verifier_indices_dict(verifier_indices_dict)

    # Save non-verifier indices
    non_verifier_indices_dict_path = os.path.join(generalization_experiment_results_dir, "non_verifier_indices.json")
    with open(non_verifier_indices_dict_path, 'w') as f:
        json.dump(non_verifier_indices_dict, f)

    num_times_compute_fid = 30

    fid_score_dict = {}

    # Loop through digits
    for digit in digit_array:

        verifier_indices = verifier_indices_dict[f"{digit}"]
        non_verifier_indices = non_verifier_indices_dict[f"{digit}"]
        generated_samples_dir = os.path.join(generalization_experiment_results_dir, f"digit_{digit}_generated_samples")

        # Compute FID Score
        against_verifier_data_fid, against_non_verifier_data_fid, all_measured_against_verifier_data_fids, all_measured_against_non_verifier_data_fids = compute_fid_score(
            fid_model_location=fid_model_location,
            num_times_compute_fid=num_times_compute_fid,
            num_generated_samples=n_samples,
            generated_results_dir=generated_samples_dir,
            verifier_indices=verifier_indices,
            non_verifier_indices=non_verifier_indices,
            device=device
        )

        print(f"Against Verifier Data FID Score for digit {digit}: {against_verifier_data_fid}")
        print(f"Against Non-Verifier Data FID Score for digit {digit}: {against_non_verifier_data_fid}")

        fid_score_dict[digit] = {
            "mean fid score against verifier data": against_verifier_data_fid,
            "mean fid score against non-verifier data": against_non_verifier_data_fid,
            "all measured against verifier data fids": all_measured_against_verifier_data_fids,
            "all measured against non-verifier data fids": all_measured_against_non_verifier_data_fids
        }

    # Save FID score dict to json
    fid_score_dict_path = os.path.join(generalization_experiment_results_dir, "fid_scores.json")
    with open(fid_score_dict_path, 'w') as f:
        json.dump(fid_score_dict, f, indent=4)
    print(f"FID scores saved to {fid_score_dict_path}")

if __name__ == '__main__':
    main()