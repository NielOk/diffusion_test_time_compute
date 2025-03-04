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

# --- New: set which "type" of model and whether to use EMA weights ---
MODEL_TYPE = "nlc"  # "lc" or "nlc" (you said you'll always use non-label-conditioned, so "nlc" is default)
USE_EMA = True      # If True, load ckpt["model_ema"], else load ckpt["model"]

# Path to the trained unconditional diffusion model checkpoint
CHECKPOINT = "epoch_100_steps_00046900.pt"

# For “search pruning” – how often we prune candidates
CHECKPOINTS = [100, 300, 600, 700, 800, 900]
APPROACHES_TO_TRY = ["mse", "bayes", "mixture"]
N_EXPERIMENTS_PER_DIGIT = 1
VERIFIER_DATA_SIZES = [10, 50, 100, 150, 200, 250, 300, 350, 400]

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
# Global Helper / Cache
# ============================
_distribution_cache = {}  # key: (digit, subset_size, approach) -> precomputed_data


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
        np.random.shuffle(indices)
        indices = indices[:subset_size]

    subset = Subset(full_dataset, indices)

    if DEBUG_VISUALIZE_DIGIT and len(indices) > 0:
        sample_image, sample_label = full_dataset[indices[0]]
        plt.imshow(sample_image.squeeze(), cmap='gray')
        plt.title(f"Sample Digit: {sample_label}")
        plt.show()

    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def load_diffusion_model(ckpt_path, device="cuda", use_ema=False):
    """
    Load your trained DDPM model (unconditional).
    Now has a 'use_ema' toggle to pick which part of the state dict to load.
    """
    model = MNISTDiffusion(
        timesteps=1000,
        image_size=28,
        in_channels=1,
        base_dim=64,
        dim_mults=[2, 4]
    )
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if use_ema:
        # Load EMA weights
        model.load_state_dict(ckpt["model_ema"], strict=False)
        print("Loaded EMA weights from checkpoint.")
    else:
        # Load standard model weights
        model.load_state_dict(ckpt["model"], strict=False)
        print("Loaded non-EMA weights from checkpoint.")

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
# Distribution Estimation
# ============================
def estimate_target_distribution_mse(model, digit_loader, digit, checkpoints, device="cuda"):
    """
    MSE-based: returns a dict {t: (1,1,28,28)} with the mean image scaled by sqrt_alphas_cumprod[t].
    """
    sum_x0 = 0.0
    count = 0
    for batch in digit_loader:
        images, labels = batch
        assert (labels == digit).all(), "Digit mismatch!"
        imgs = images.to(device)
        sum_x0 += imgs.sum(dim=0)
        count += imgs.shape[0]

    global_mean = sum_x0 / count  # shape [1,28,28]

    target_means = {}
    for t in checkpoints:
        factor = model.sqrt_alphas_cumprod[t]
        target_means[t] = (factor * global_mean).unsqueeze(0)  # shape [1,1,28,28]
    return target_means


def estimate_target_distribution_bayes(model, digit_loader, digit, checkpoints, device="cuda"):
    """
    Bayesian single-Gaussian approach.
    p(x_t) = Product_i N(x_t; sqrt_alpha * x0_i, (1-alpha) I).
    => yields single Gaussian, mean= average(sqrt_alpha * x0_i), var= (1-alpha)/N
    We'll store (posterior_means[t], posterior_vars[t]).
    """
    sum_x0 = 0.0
    count = 0
    for batch in digit_loader:
        images, labels = batch
        assert (labels == digit).all(), "Digit mismatch!"
        imgs = images.to(device)
        sum_x0 += imgs.sum(dim=0)
        count += imgs.shape[0]

    global_mean_x0 = sum_x0 / count  # shape [1,28,28]

    posterior_means = {}
    posterior_vars = {}

    for t in checkpoints:
        alpha_t = model.alphas_cumprod[t]
        sqrt_alpha = torch.sqrt(alpha_t)
        mu_t = sqrt_alpha * global_mean_x0
        var_t = (1.0 - alpha_t.item()) / float(count)

        posterior_means[t] = mu_t.unsqueeze(0)  # [1,1,28,28]
        posterior_vars[t]  = var_t

    return (posterior_means, posterior_vars)


def estimate_target_distribution_mixture(model, digit_loader, digit, checkpoints, device="cuda"):
    """
    Mixture approach:
      p(x_t) = (1/N) sum_{i=1..N} N(x_t; sqrt_alpha*x0_i, (1-alpha)*I).
    We'll store a dict: mixture_data[t] = (mus, var_t)
      mus = [N,1,28,28], var_t float
    """
    all_x0 = []
    for batch in digit_loader:
        images, labels = batch
        assert (labels == digit).all(), "Digit mismatch!"
        imgs = images.to(device)
        all_x0.append(imgs)
    all_x0 = torch.cat(all_x0, dim=0)  # shape [N,1,28,28]
    count = all_x0.shape[0]

    mixture_data = {}
    for t in checkpoints:
        alpha_t = model.alphas_cumprod[t]
        sqrt_alpha = torch.sqrt(alpha_t)
        var_t = (1.0 - alpha_t).item()
        mus = sqrt_alpha * all_x0  # shape [N,1,28,28]
        mixture_data[t] = (mus, var_t)

    return mixture_data


# ============================
# Scoring / Branch-and-Prune
# ============================
def reverse_diffusion_search_mse(model, target_means, n_candidates=64, device="cuda"):
    """
    MSE-based branch-and-prune. target_means: dict {t: [1,1,28,28]}
    """
    candidates = torch.randn((n_candidates, model.in_channels, model.image_size, model.image_size),
                             device=device)

    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)

        if t in target_means:
            target = target_means[t].expand(candidates.shape[0], -1, -1, -1)
            scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
            k = max(1, candidates.shape[0] // 2)
            topk_indices = torch.topk(scores, k=k).indices
            candidates = candidates[topk_indices]

    if 0 in target_means:
        target_final = target_means[0].expand(candidates.shape[0], -1, -1, -1)
        final_scores = -F.mse_loss(candidates, target_final, reduction='none').mean(dim=[1,2,3])
        best_idx = torch.argmax(final_scores)
        best_candidate = candidates[best_idx]
    else:
        best_candidate = candidates[0]

    return best_candidate


def reverse_diffusion_search_bayes(model, bayes_data, n_candidates=64, device="cuda"):
    """
    Bayes single-Gaussian negative log-likelihood approach.
    bayes_data = (posterior_means, posterior_vars).
    """
    posterior_means, posterior_vars = bayes_data

    candidates = torch.randn((n_candidates, model.in_channels, model.image_size, model.image_size),
                             device=device)

    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)

        if t in posterior_means:
            mu_t = posterior_means[t].expand(candidates.shape[0], -1, -1, -1)
            var_t = posterior_vars[t]
            squared_errors = F.mse_loss(candidates, mu_t, reduction='none').mean(dim=[1,2,3])
            scores = -squared_errors / (2.0 * var_t)
            k = max(1, candidates.shape[0] // 2)
            topk_indices = torch.topk(scores, k=k).indices
            candidates = candidates[topk_indices]

    if 0 in posterior_means:
        mu_0 = posterior_means[0].expand(candidates.shape[0], -1, -1, -1)
        var_0 = posterior_vars[0]
        squared_errors_0 = F.mse_loss(candidates, mu_0, reduction='none').mean(dim=[1,2,3])
        final_scores = -squared_errors_0 / (2.0 * var_0)
        best_idx = torch.argmax(final_scores)
        best_candidate = candidates[best_idx]
    else:
        best_candidate = candidates[0]

    return best_candidate


def reverse_diffusion_search_mixture(model, mixture_data, n_candidates=64, device="cuda"):
    """
    Mixture negative log-likelihood approach.
    mixture_data[t] = (mus, var_t) with mus shape [N,1,28,28].
    """
    candidates = torch.randn((n_candidates, model.in_channels, model.image_size, model.image_size),
                             device=device)

    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)

        if t in mixture_data:
            mus, var_t = mixture_data[t]
            # Evaluate log-likelihood under the mixture for each candidate
            batch_size = candidates.shape[0]
            c_expanded = candidates.unsqueeze(1)  # [batch,1,1,28,28]
            m_expanded = mus.unsqueeze(0)         # [1,N,1,28,28]
            diff = c_expanded - m_expanded
            sq_dist = diff.pow(2).mean(dim=[2,3,4])  # [batch, N]
            exponent = -sq_dist / (2.0 * var_t)
            max_exponent, _ = exponent.max(dim=1, keepdim=True)
            exponent_shifted = exponent - max_exponent
            sum_exp = torch.exp(exponent_shifted).sum(dim=1)
            mixture_sum = (1.0 / mus.shape[0]) * torch.exp(max_exponent.squeeze()) * sum_exp
            ll = torch.log(mixture_sum + 1e-12)
            scores = ll

            k = max(1, batch_size // 2)
            topk_indices = torch.topk(scores, k=k).indices
            candidates = candidates[topk_indices]

    # Final pick for t=0 if available
    if 0 in mixture_data:
        mus_0, var_0 = mixture_data[0]
        batch_size = candidates.shape[0]
        c_expanded = candidates.unsqueeze(1)
        m_expanded = mus_0.unsqueeze(0)
        diff = c_expanded - m_expanded
        sq_dist = diff.pow(2).mean(dim=[2,3,4])  # [batch, N]
        exponent = -sq_dist / (2.0 * var_0)
        max_exponent, _ = exponent.max(dim=1, keepdim=True)
        exponent_shifted = exponent - max_exponent
        sum_exp = torch.exp(exponent_shifted).sum(dim=1)
        mixture_sum = (1.0 / mus_0.shape[0]) * torch.exp(max_exponent.squeeze()) * sum_exp
        ll = torch.log(mixture_sum + 1e-12)
        best_idx = torch.argmax(ll)
        best_candidate = candidates[best_idx]
    else:
        best_candidate = candidates[0]

    return best_candidate


# ============================
# Dispatchers
# ============================
def get_distribution_for_digit(model, digit_loader, digit, checkpoints, device, approach):
    """Select the distribution-estimation function based on 'approach'."""
    if approach == "mse":
        return estimate_target_distribution_mse(model, digit_loader, digit, checkpoints, device)
    elif approach == "bayes":
        return estimate_target_distribution_bayes(model, digit_loader, digit, checkpoints, device)
    elif approach == "mixture":
        return estimate_target_distribution_mixture(model, digit_loader, digit, checkpoints, device)
    else:
        raise ValueError(f"Unknown approach: {approach}")


def perform_search(model, distribution_data, approach, n_candidates, device):
    """Dispatch to the correct search method based on 'approach'."""
    if approach == "mse":
        return reverse_diffusion_search_mse(model, distribution_data, n_candidates, device)
    elif approach == "bayes":
        return reverse_diffusion_search_bayes(model, distribution_data, n_candidates, device)
    elif approach == "mixture":
        return reverse_diffusion_search_mixture(model, distribution_data, n_candidates, device)
    else:
        raise ValueError(f"Unknown approach: {approach}")


# ============================
# Main Experiment Routines
# ============================
def generate_samples_for_digit(
    model,
    digit,
    verifier_data_subset_size,
    search_method_name="mse",
    n_experiments=10,
    checkpoints=CHECKPOINTS,
    n_candidates=N_CANDIDATES,
    device="cuda"
):
    """
    1) Load data for that digit (subset_size).
    2) Compute/fetch distribution for 'search_method_name'.
    3) Perform generation n_experiments times.
    4) Return list of (image_tensor, intended_digit).
    """
    dist_key = (digit, verifier_data_subset_size, search_method_name)
    if dist_key not in _distribution_cache:
        print(f"  -> Computing distribution for digit={digit}, subset_size={verifier_data_subset_size}, approach={search_method_name}")
        digit_loader = create_digit_dataloader(digit=digit, subset_size=verifier_data_subset_size)
        dist_data = get_distribution_for_digit(
            model=model,
            digit_loader=digit_loader,
            digit=digit,
            checkpoints=checkpoints,
            device=device,
            approach=search_method_name
        )
        _distribution_cache[dist_key] = dist_data
    else:
        dist_data = _distribution_cache[dist_key]
        print(f"  -> Using cached distribution for digit={digit}, subset_size={verifier_data_subset_size}, approach={search_method_name}")

    out_samples = []
    for exp_idx in range(n_experiments):
        print(f"    [Digit {digit}] Generating sample {exp_idx+1}/{n_experiments} with approach={search_method_name}")
        best_noise = perform_search(
            model=model,
            distribution_data=dist_data,
            approach=search_method_name,
            n_candidates=n_candidates,
            device=device
        )
        out_samples.append((best_noise.unsqueeze(0), digit))

    return out_samples


def run_scaling_study(
    diffusion_model,
    hf_model,
    feature_extractor,
    verifier_data_sizes=[10, 50, 100],
    search_method_name="mse",
    n_experiments_per_digit=3,
    device="cuda"
):
    """
    Evaluate classification accuracy for different subset sizes.
    Returns { subset_size : accuracy }.
    """
    results = {}

    for subset_size in verifier_data_sizes:
        print(f"\n=== Running experiments for subset_size={subset_size}, approach={search_method_name} ===")

        generated_samples = []
        for digit in DIGIT_ARRAY:
            digit_samples = generate_samples_for_digit(
                model=diffusion_model,
                digit=digit,
                verifier_data_subset_size=subset_size,
                search_method_name=search_method_name,
                n_experiments=n_experiments_per_digit,
                device=device
            )
            generated_samples.extend(digit_samples)

        # Classify
        all_images = torch.cat([item[0] for item in generated_samples], dim=0)
        all_intended_digits = np.array([item[1] for item in generated_samples], dtype=int)

        predicted_labels = classify_generated_images_hf(all_images, hf_model, feature_extractor, device=device)

        correct = np.sum(predicted_labels == all_intended_digits)
        total = len(all_intended_digits)
        accuracy = correct / total

        print(f"  => {search_method_name.upper()} | subset_size={subset_size} | Accuracy={accuracy*100:.2f}% ({correct}/{total})")

        results[subset_size] = accuracy

        # Optionally: confusion matrix
        cm = confusion_matrix(all_intended_digits, predicted_labels, labels=list(range(10)))
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted")
        plt.ylabel("Intended")
        plt.title(f"Confusion Matrix (Subset={subset_size}, Approach={search_method_name})")
        cm_path = os.path.join(LOG_DIR, f"cm_{search_method_name}_subset_{subset_size}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()

    return results


def plot_scaling_study_results(results_dict, approach="mse"):
    """
    Plot accuracy vs. subset_size from results_dict = {subset_size: accuracy}
    """
    sizes = sorted(results_dict.keys())
    accuracies = [results_dict[s] for s in sizes]

    plt.figure()
    plt.plot(sizes, accuracies, marker='o')
    plt.xlabel("Number of Labeled Examples (per digit) for Verifier")
    plt.ylabel("Classification Accuracy")
    plt.title(f"{approach.upper()} Approach: Accuracy vs. Verifier Data Size")
    plot_path = os.path.join(LOG_DIR, f"scaling_study_accuracy_{approach}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved scaling study plot to {plot_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, CHECKPOINT)

    print(f"Loading diffusion model from: {ckpt_path}")
    diffusion_model = load_diffusion_model(
        ckpt_path,
        device=device,
        use_ema=USE_EMA  # <- Pass in our new toggle
    )

    # Load Hugging Face classifier
    hf_model, feature_extractor = load_hf_classifier(device=device)

    # Approaches to try
    approaches_to_try = APPROACHES_TO_TRY

    # Subset sizes for the scaling study:
    verifier_data_sizes = VERIFIER_DATA_SIZES
    n_experiments_per_digit = N_EXPERIMENTS_PER_DIGIT
    
    # We'll create one log file for *all* runs
    full_log_path = os.path.join(LOG_DIR, "combined_experiment_log.txt")
    with open(full_log_path, "w") as f_log:
        # Write high-level config
        f_log.write("=== EXPERIMENT DETAILS ===\n")
        f_log.write(f"Timestamp: {timestamp}\n")
        f_log.write(f"MODEL_TYPE: {MODEL_TYPE}\n")
        f_log.write(f"USE_EMA: {USE_EMA}\n")
        f_log.write(f"Checkpoints: {CHECKPOINTS}\n")
        f_log.write(f"N_CANDIDATES: {N_CANDIDATES}\n")
        f_log.write(f"verifier_data_sizes: {verifier_data_sizes}\n")
        f_log.write(f"n_experiments_per_digit: {n_experiments_per_digit}\n")
        f_log.write(f"Diffusion Model (NLC) Checkpoint: {CHECKPOINT}\n")
        f_log.write(f"HF_MODEL_NAME: {HF_MODEL_NAME}\n\n")

        for approach in approaches_to_try:
            print(f"\n======== Starting scaling study for approach: {approach} ========\n")
            f_log.write(f"=== Approach: {approach} ===\n")

            # Run scaling study
            results = run_scaling_study(
                diffusion_model=diffusion_model,
                hf_model=hf_model,
                feature_extractor=feature_extractor,
                verifier_data_sizes=verifier_data_sizes,
                search_method_name=approach,
                n_experiments_per_digit=n_experiments_per_digit,
                device=device
            )

            # Plot results
            plot_scaling_study_results(results, approach=approach)

            # Save a separate text summary for just this approach
            approach_result_path = os.path.join(LOG_DIR, f"results_{approach}.txt")
            with open(approach_result_path, "w") as f_approach:
                f_approach.write(f"Approach: {approach}\n")
                f_approach.write(f"Checkpoints: {CHECKPOINTS}\n")
                f_approach.write(f"N_CANDIDATES: {N_CANDIDATES}\n")
                f_approach.write(f"verifier_data_sizes: {verifier_data_sizes}\n")
                f_approach.write(f"n_experiments_per_digit: {n_experiments_per_digit}\n")
                f_approach.write("Per-subset accuracy results:\n")
                for sz, acc in results.items():
                    f_approach.write(f"  subset_size={sz}, accuracy={acc:.4f}\n")

            # Also append to the combined log file
            f_log.write(f"Approach: {approach}\n")
            for sz, acc in results.items():
                f_log.write(f"  subset_size={sz}, accuracy={acc:.4f}\n")
            f_log.write("\n")

    print(f"\nAll experiments complete. Detailed logs saved to {full_log_path}")


if __name__ == "__main__":
    main()
