#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # If running on a headless server
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

# ============================
# Script Configuration
# ============================

# Path to the trained unconditional diffusion model checkpoint
CHECKPOINT = "epoch_100_steps_00046900.pt"

# For “search pruning” – how often we prune candidates
CHECKPOINTS = [100, 400, 500, 600, 700, 800]

# Number of noise candidates to spawn at each attempt
N_CANDIDATES = 64

# Hugging Face MNIST classifier repository
HF_MODEL_NAME = "farleyknight/mnist-digit-classification-2022-09-04"

# Data
MNIST_ROOT = "./mnist_data"
DEBUG_VISUALIZE_DIGIT = False  # If True, will show a sample digit from the subset

# Where this script resides
MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'nlc_gpu_accelerated_training')
TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'nlc_trained_ddpm', 'results')

DIGIT_ARRAY = list(range(10))

sys.path.append(GPU_ACCELERATED_TRAINING_DIR)
sys.path.append(TRAINED_MODELS_DIR)

from model import MNISTDiffusion  # Adjust if your model import path is different

# ============================
# Logging Directory
# ============================

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(MODEL_TEST_DIR, f"experiment_outputs_{timestamp}")
os.makedirs(LOG_DIR, exist_ok=True)
print(f"All logs & figures will be saved to: {LOG_DIR}")

# ============================
# Helper Functions
# ============================

def create_digit_dataloader(digit, subset_size=None, batch_size=128, image_size=28, num_workers=0):
    """
    Load MNIST samples for a specific digit.
    If subset_size is given, randomly select that many samples for the digit (simulating limited data).
    """
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Range: [-1, 1]
    ])
    full_dataset = datasets.MNIST(root=MNIST_ROOT, train=True, download=True, transform=preprocess)
    # Filter for the requested digit
    indices = [i for i, target in enumerate(full_dataset.targets) if target == digit]

    if subset_size is not None and subset_size < len(indices):
        # Randomly pick a subset of these indices
        np.random.shuffle(indices)
        indices = indices[:subset_size]

    subset = Subset(full_dataset, indices)

    if DEBUG_VISUALIZE_DIGIT and len(indices) > 0:
        sample_image, sample_label = full_dataset[indices[0]]
        plt.imshow(sample_image.squeeze(), cmap='gray')
        plt.title(f"Sample Digit: {sample_label}")
        plt.show()

    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_diffusion_model(ckpt_path, device="cuda"):
    """Load your trained DDPM model (unconditional)."""
    model = MNISTDiffusion(
        timesteps=1000,
        image_size=28,
        in_channels=1,
        base_dim=64,
        dim_mults=[2, 4]
    )
    model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def load_hf_classifier(model_name=HF_MODEL_NAME, device="cuda"):
    """Loads a Hugging Face MNIST classifier (e.g. a ViT model)."""
    print(f"Loading Hugging Face model: {model_name}")
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model.eval()
    return model, feature_extractor


def classify_generated_images_hf(torch_images, hf_model, feature_extractor, device="cuda"):
    """
    Classify a batch of PyTorch images (shape [N,1,28,28], range ~[-1,1]) using a Hugging Face model.
    Returns predicted labels as a numpy array of shape [N].
    """
    from PIL import Image

    pil_images = []
    for i in range(torch_images.size(0)):
        arr = torch_images[i].detach().cpu().numpy().squeeze()
        arr_255 = (arr + 1.0) / 2.0 * 255.0
        arr_255 = np.clip(arr_255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr_255, mode='L').convert('RGB')
        pil_images.append(pil_img)

    inputs = feature_extractor(images=pil_images, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    with torch.no_grad():
        logits = hf_model(**inputs).logits  # shape [N,10]
    predicted_labels = logits.argmax(dim=-1).cpu().numpy()
    return predicted_labels


# ============================
# Search / Verifier Interface
# ============================
# We define a modular “verifier” interface and a search function so that in future
# you can swap in different approaches besides MSE-based.

def estimate_target_distribution_mse(model, digit_loader, digit, checkpoints, device="cuda"):
    """
    Compute a global mean of real images for a given digit and map it onto different timesteps,
    for use as an MSE-based target distribution.
    """
    sum_x0 = 0.0
    count = 0
    for batch in digit_loader:
        images, labels = batch
        # Basic check that all labels match
        assert (labels == digit).all(), "Digit loader contains incorrect labels!"
        imgs = images.to(device)
        sum_x0 += imgs.sum(dim=0)
        count += imgs.shape[0]

    global_mean = sum_x0 / count  # shape [1, 28, 28]

    # For each checkpoint t, store factor * mean (where factor = sqrt_alphas_cumprod[t])
    target_means = {}
    for t in checkpoints:
        factor = model.sqrt_alphas_cumprod[t]
        target_means[t] = factor * global_mean.unsqueeze(0)  # shape [1, 1, 28, 28]
    return target_means


def reverse_diffusion_search_mse(model, target_means, n_candidates=64, device="cuda"):
    """
    Example MSE-based branch-and-prune search using precomputed target_means at certain timesteps.
    """
    candidates = torch.randn((n_candidates, model.in_channels, model.image_size, model.image_size),
                             device=device)

    # Sort the timesteps in descending order just to be sure
    all_timesteps = sorted(target_means.keys(), reverse=True)

    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)

        # If t is in target_means, we do a pruning step
        if t in target_means:
            target = target_means[t].expand(candidates.shape[0], -1, -1, -1)
            scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
            # Keep top half of the candidates
            k = max(1, candidates.shape[0] // 2)
            topk_indices = torch.topk(scores, k=k).indices
            candidates = candidates[topk_indices]

    # Optionally pick the single best vs. t=0 target
    if 0 in target_means:
        target_final = target_means[0].expand(candidates.shape[0], -1, -1, -1)
        final_scores = -F.mse_loss(candidates, target_final, reduction='none').mean(dim=[1,2,3])
        best_idx = torch.argmax(final_scores)
        best_candidate = candidates[best_idx]
    else:
        best_candidate = candidates[0]

    return best_candidate


# We'll create a dictionary of possible search methods. Right now we only have "mse_branch_prune".
SEARCH_METHODS = {
    "mse_branch_prune": reverse_diffusion_search_mse,
}


# ============================
# Main Experiment Routines
# ============================

def generate_samples_for_digit(
    model,
    digit,
    verifier_data_subset_size,
    search_method_name,
    n_experiments=10,
    checkpoints=CHECKPOINTS,
    n_candidates=N_CANDIDATES,
    device="cuda"
):
    """
    For a single digit, do:
      1) Create dataloader with 'verifier_data_subset_size' samples
      2) Estimate MSE-based target distribution (or a different approach, if you code it)
      3) Use the chosen search method to generate images
      4) Return list of (image_tensor, intended_digit)
    """
    # 1) Build dataloader for the digit (subset size = verifier_data_subset_size)
    digit_loader = create_digit_dataloader(digit=digit, subset_size=verifier_data_subset_size)

    # 2) Estimate target distribution means (for MSE approach)
    #    In the future, you could have if-else logic for different verifiers
    target_means = estimate_target_distribution_mse(model, digit_loader, digit, checkpoints, device)

    # 3) Perform generation
    search_fn = SEARCH_METHODS[search_method_name]  # e.g. reverse_diffusion_search_mse
    out_samples = []
    for _ in range(n_experiments):
        best_noise = search_fn(model, target_means, n_candidates=n_candidates, device=device)
        out_samples.append((best_noise.unsqueeze(0), digit))  # shape [1,1,28,28], digit
    return out_samples


def run_scaling_study(
    diffusion_model,
    hf_model,
    feature_extractor,
    verifier_data_sizes=[10, 50, 100, 500, 1000],
    search_method_name="mse_branch_prune",
    n_experiments_per_digit=10,
    device="cuda"
):
    """
    Runs a set of experiments to see how classification accuracy depends on how many
    labeled examples the verifier sees for each digit.
    Returns a dictionary: { data_size : accuracy }, plus we save relevant logs/plots.
    """
    results = {}

    for subset_size in verifier_data_sizes:
        print(f"\n=== Running experiments with verifier_data_subset_size={subset_size} ===")

        # Collect all generated samples across digits
        generated_samples = []
        for digit in DIGIT_ARRAY:
            digit_samples = generate_samples_for_digit(
                model=diffusion_model,
                digit=digit,
                verifier_data_subset_size=subset_size,
                search_method_name=search_method_name,
                n_experiments= n_experiments_per_digit,
                device=device
            )
            generated_samples.extend(digit_samples)

        # Classify
        all_images = torch.cat([item[0] for item in generated_samples], dim=0)  # shape: [N,1,28,28]
        all_intended_digits = np.array([item[1] for item in generated_samples], dtype=int)

        predicted_labels = classify_generated_images_hf(all_images, hf_model, feature_extractor, device=device)

        correct = np.sum(predicted_labels == all_intended_digits)
        total = len(all_intended_digits)
        accuracy = correct / total
        print(f"Verifier subset_size={subset_size} => Accuracy: {accuracy*100:.2f}%  ({correct}/{total})")

        results[subset_size] = accuracy

        # Optionally, you can save out confusion matrices here if you like
        cm = confusion_matrix(all_intended_digits, predicted_labels, labels=list(range(10)))
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted")
        plt.ylabel("Intended")
        plt.title(f"Confusion Matrix (Verifier subset size={subset_size})")
        cm_path = os.path.join(LOG_DIR, f"cm_subset_{subset_size}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()

    return results


def plot_scaling_study_results(results_dict, title="Scaling Study: Accuracy vs. Verifier Data Size"):
    """
    Given a dict { data_size : accuracy }, produce a line plot and save it out.
    """
    sizes = sorted(results_dict.keys())
    accuracies = [results_dict[s] for s in sizes]

    plt.figure()
    plt.plot(sizes, accuracies, marker='o')
    plt.xlabel("Number of Labeled Examples (per digit) for Verifier")
    plt.ylabel("Classification Accuracy")
    plt.title(title)
    scale_plot_path = os.path.join(LOG_DIR, "scaling_study_accuracy.png")
    plt.savefig(scale_plot_path, dpi=150)
    plt.close()
    print(f"Saved scaling study plot to {scale_plot_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, CHECKPOINT)

    print(f"Loading diffusion model from: {ckpt_path}")
    diffusion_model = load_diffusion_model(ckpt_path, device=device)

    # Load Hugging Face classifier
    hf_model, feature_extractor = load_hf_classifier(device=device)

    # Define the different subset sizes to test. You can adjust these as desired.
    verifier_data_sizes = [10, 1000, 2000]

    # We’ll do 10 “experiments” per digit (like your original N_EXPERIMENTS).
    n_experiments_per_digit = 10

    # For now, we only have the “mse_branch_prune” method, but you can add more.
    results = run_scaling_study(
        diffusion_model=diffusion_model,
        hf_model=hf_model,
        feature_extractor=feature_extractor,
        verifier_data_sizes=verifier_data_sizes,
        search_method_name="mse_branch_prune",
        n_experiments_per_digit=n_experiments_per_digit,
        device=device
    )

    # Plot final scaling results
    plot_scaling_study_results(results, title="MSE Branch-and-Prune Accuracy vs. Verifier Data Size")

    # Also log results to a text file
    results_path = os.path.join(LOG_DIR, "scaling_study_results.txt")
    with open(results_path, "w") as f:
        for sz, acc in results.items():
            f.write(f"subset_size={sz}, accuracy={acc:.4f}\n")
    print(f"Saved scaling study text results to {results_path}")


if __name__ == "__main__":
    main()
