#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import datetime

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Hugging Face imports
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

# Metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ============================
# Script Configuration
# ============================

# Path to the trained unconditional diffusion model checkpoint
CHECKPOINT = "epoch_100_steps_00046900.pt"

# Number of experiments per digit (how many times we steer the model for each digit)
N_EXPERIMENTS = 5

# For “search pruning” – how often we prune candidates
SCORING_TIMESTEPS = 200
CHECKPOINTS = [100, 400, 500, 600, 700, 800]

# Number of noise candidates to spawn at each attempt
N_CANDIDATES = 64

# Hugging Face MNIST classifier repository
# (For example: "farleyknight/mnist-digit-classification-2022-09-04")
HF_MODEL_NAME = "farleyknight/mnist-digit-classification-2022-09-04"

# Data
MNIST_ROOT = "./mnist_data"
DEBUG_VISUALIZE_DIGIT = False  # If True, will show a sample digit from the subset

# Where this script resides
MODEL_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(MODEL_TEST_DIR)
GPU_ACCELERATED_TRAINING_DIR = os.path.join(REPO_DIR, 'nlc_gpu_accelerated_training')
TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'nlc_trained_ddpm', 'results')

DIGIT_ARRAY = [5]

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

def create_digit_dataloader(digit, batch_size=128, image_size=28, num_workers=0):
    """Load all MNIST samples for a specific digit, for estimating that digit's global mean."""
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Range: [-1, 1]
    ])
    full_dataset = datasets.MNIST(root=MNIST_ROOT, train=True, download=True, transform=preprocess)
    indices = [i for i, target in enumerate(full_dataset.targets) if target == digit]
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


def estimate_target_distribution(model, digit_loader, digit, checkpoints, device="cuda"):
    """Compute the mean of real images for a given digit and map it onto different timesteps."""
    sum_x0 = 0.0
    count = 0
    for batch in digit_loader:
        images, labels = batch
        # Safety check to confirm all labels match the requested digit
        assert (labels == digit).all(), "Digit loader contains incorrect labels!"
        imgs = images.to(device)
        sum_x0 += imgs.sum(dim=0)
        count += imgs.shape[0]

    # Mean image for that digit (shape: [1, 1, 28, 28])
    global_mean = sum_x0 / count

    # For each checkpoint t, store factor * mean (where factor = sqrt_alphas_cumprod[t])
    target_means = {}
    for t in checkpoints:
        factor = model.sqrt_alphas_cumprod[t]
        # Expand dims to match (batch, C, H, W)
        target_means[t] = factor * global_mean.unsqueeze(0)
    return target_means


def reverse_diffusion_search(model, target_means, n_candidates=N_CANDIDATES,
                             scoring_interval=SCORING_TIMESTEPS, device="cuda"):
    """
    - Start with n_candidates noise samples.
    - Reverse diffuse from t=999 down to t=0.
    - At each scoring checkpoint in 'target_means', select the top K (half) by MSE score.
    - Return the single best sample at the end.
    """
    candidates = torch.randn((n_candidates, model.in_channels, model.image_size, model.image_size), device=device)

    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)

        # Prune candidates if we're at a designated checkpoint
        if t in target_means:
            target = target_means[t].expand(candidates.shape[0], -1, -1, -1)
            # We'll use negative MSE so that a higher "score" is better
            scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
            k = max(1, candidates.shape[0] // 2)
            topk_indices = torch.topk(scores, k=k).indices
            candidates = candidates[topk_indices]

    # Among the remaining candidates, pick the single best vs. target_means[0] if available
    if 0 in target_means:
        target_final = target_means[0].expand(candidates.shape[0], -1, -1, -1)
        final_scores = -F.mse_loss(candidates, target_final, reduction='none').mean(dim=[1,2,3])
        best_idx = torch.argmax(final_scores)
        best_candidate = candidates[best_idx]
    else:
        # If for some reason t=0 wasn't in target_means, just pick the first
        best_candidate = candidates[0]

    return best_candidate


def plot_image(image_tensor, title="Generated Image"):
    """Convert [-1,1] image tensor to [0,1] range for plotting."""
    image = image_tensor.squeeze().detach().cpu().numpy()
    image_norm = (image + 1) / 2.0  # from [-1, 1] to [0, 1]
    plt.imshow(image_norm, cmap='gray')
    plt.title(title)
    plt.axis('off')


# ======== Hugging Face Classifier Loading & Inference =========

def load_hf_classifier(model_name=HF_MODEL_NAME, device="cuda"):
    """
    Loads a Hugging Face MNIST classifier (e.g. a ViT model).
    """
    print(f"Loading Hugging Face model: {model_name}")
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model.eval()
    return model, feature_extractor

def classify_generated_images_hf(torch_images, hf_model, feature_extractor, device="cuda"):
    """
    Classify a batch of PyTorch images (shape [N,1,28,28], range ~[-1,1])
    using the Hugging Face model.
    Returns predicted labels as a numpy array of shape [N].
    """
    from PIL import Image

    pil_images = []
    for i in range(torch_images.size(0)):
        # Convert each image to CPU array
        arr = torch_images[i].detach().cpu().numpy().squeeze()
        # Scale [-1,1] => [0,255]
        arr_255 = (arr + 1.0) / 2.0 * 255.0
        arr_255 = np.clip(arr_255, 0, 255).astype(np.uint8)

        # Create a grayscale PIL image, then convert to RGB
        pil_img = Image.fromarray(arr_255, mode='L').convert('RGB')
        pil_images.append(pil_img)

    # Let the feature extractor handle resizing, normalization, etc.
    # We'll feed all images at once if memory allows; otherwise, chunk them.
    inputs = feature_extractor(images=pil_images, return_tensors="pt")

    # Move input tensors to the correct device
    for k, v in inputs.items():
        inputs[k] = v.to(device)

    # Forward pass
    with torch.no_grad():
        logits = hf_model(**inputs).logits  # shape [N,10]
    predicted_labels = logits.argmax(dim=-1).cpu().numpy()
    return predicted_labels


# ============================
# Main Experiment
# ============================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, CHECKPOINT)

    print(f"Loading diffusion model from: {ckpt_path}")
    diffusion_model = load_diffusion_model(ckpt_path, device)

    # Load Hugging Face classifier
    hf_model, feature_extractor = load_hf_classifier(device=device)

    # We’ll store all generated images here for final evaluation
    # Each element: (tensor_of_shape[1,1,28,28], intended_digit)
    generated_samples = []

    # Decide which timesteps we’ll compute target_means for
    checkpoints = CHECKPOINTS
    print(f"Scoring checkpoints: {checkpoints}")

    # For final montage
    fig_montage = plt.figure(figsize=(15, 8))
    fig_montage.suptitle("Generated Samples (Best Candidate per Experiment)")

    # Generate images for digits 0-9
    for digit in DIGIT_ARRAY:
        # 1) Build dataloader for the digit
        digit_loader = create_digit_dataloader(digit=digit, batch_size=128)

        # 2) Estimate target distribution means
        target_means = estimate_target_distribution(diffusion_model, digit_loader, digit, checkpoints, device)

        # 3) Generate N_EXPERIMENTS images by steering the diffusion model
        for experiment_idx in range(N_EXPERIMENTS):
            best_noise = reverse_diffusion_search(
                diffusion_model,
                target_means,
                n_candidates=N_CANDIDATES,
                scoring_interval=SCORING_TIMESTEPS,
                device=device
            )
            # Store result (add a batch dimension so shape is [1,1,28,28])
            generated_samples.append((best_noise.unsqueeze(0), digit))
            print(f"[Digit {digit}] Experiment {experiment_idx+1}/{N_EXPERIMENTS} complete.")

            # Plot in the montage
            subplot_idx = digit * N_EXPERIMENTS + experiment_idx + 1
            ax = fig_montage.add_subplot(10, N_EXPERIMENTS, subplot_idx)
            image_np = best_noise.squeeze().detach().cpu().numpy()
            image_np = (image_np + 1) / 2.0
            ax.imshow(image_np, cmap='gray')
            ax.set_title(f"Digit {digit}", fontsize=8)
            ax.axis('off')

    plt.tight_layout()
    montage_path = os.path.join(LOG_DIR, "generated_samples_montage.png")
    plt.savefig(montage_path, dpi=150)
    plt.close(fig_montage)
    print(f"Saved generated samples montage to {montage_path}")

    # ==================================
    # Classification & Confusion Matrix
    # ==================================
    print("\n====> Classifying all generated samples...")

    # Combine all generated images into one tensor
    all_images = torch.cat([item[0] for item in generated_samples], dim=0)  # shape: [N,1,28,28]
    all_intended_digits = np.array([item[1] for item in generated_samples], dtype=int)

    # Use Hugging Face classifier
    predicted_labels = classify_generated_images_hf(all_images, hf_model, feature_extractor, device=device)

    # Compute accuracy
    correct = np.sum(predicted_labels == all_intended_digits)
    total = len(all_intended_digits)
    accuracy = correct / total
    print(f"Overall Classification Accuracy: {accuracy*100:.2f}%  ({correct}/{total})")

    # Save accuracy to a text file
    results_path = os.path.join(LOG_DIR, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Total Generated Samples: {total}\n")
        f.write(f"Classification Accuracy: {accuracy*100:.2f}%  ({correct}/{total})\n")
    print(f"Saved accuracy results to {results_path}")

    # Build confusion matrix
    cm = confusion_matrix(all_intended_digits, predicted_labels, labels=list(range(10)))

    # Display & save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("Intended Digit")
    plt.title("Confusion Matrix of Generated Digits vs. HF Model Predictions")

    cm_path = os.path.join(LOG_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
