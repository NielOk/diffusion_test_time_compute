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

"""
MNIST Training set sizes:
Digit 0: 5923 examples
Digit 1: 6742 examples
Digit 2: 5958 examples
Digit 3: 6131 examples
Digit 4: 5842 examples
Digit 5: 5421 examples
Digit 6: 5918 examples
Digit 7: 6265 examples
Digit 8: 5851 examples
Digit 9: 5949 examples

Confirmed that preprocessing for HF classifier is being done correctly.
"""


# ============================
# Script Configuration
# ============================

MODEL_TYPE = "nlc"  # "lc" or "nlc" – you said you'd typically use non-label-conditioned
USE_EMA = True      # If True, load ckpt["model_ema"], else ckpt["model"]

# Path to the trained unconditional diffusion model checkpoint
CHECKPOINT = "epoch_100_steps_00046900.pt"

# This list is now **only** for the top_k search:
CHECKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900]

# Approaches
# APPROACHES_TO_TRY = ["mse", "bayes", "mixture"]  # distribution approaches
# SEARCH_METHODS_TO_TRY = ["top_k", "paths"]       # search methods
APPROACHES_TO_TRY = ["mse", "mixture", "bayes"]  # distribution approaches
SEARCH_METHODS_TO_TRY = ["top_k", "paths"]  # search methods

# Number of repeated generation attempts per digit
N_EXPERIMENTS_PER_DIGIT = 50

# Subset sizes for the distribution estimation
VERIFIER_DATA_SIZES = [50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

# --- Separate numbers of candidates ---
N_CANDIDATES_TOP_K = 512
N_CANDIDATES_PATHS = 5
# --------------------------------------

# If using the "paths" method, define partial forward/backward intervals:
DELTA_F = 100 # steps to forward-diffuse
DELTA_B = 200  # steps to reverse-diffuse

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
from utils import ExponentialMovingAverage

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
# We'll now key the cache by (digit, subset_size, approach, search_method)
_distribution_cache = {}

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
    
    print(f"{len(indices)} sample images loaded for digit {digit}")

    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def load_diffusion_model(ckpt_path, device="cuda", use_ema=False):
    """
    Load the trained DDPM model, with an option to load the EMA version.
    """
    model = MNISTDiffusion(
        timesteps=1000,
        image_size=28,
        in_channels=1,
        base_dim=64,
        dim_mults=[2, 4]
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if use_ema:
        # Initialize EMA model and load EMA state_dict properly
        model_ema = ExponentialMovingAverage(model, decay=0.995, device=device)
        model_ema.load_state_dict(ckpt["model_ema"])
        model_ema.eval()  # Set to evaluation mode
        print("Loaded EMA weights from checkpoint.")
        return model_ema.module  # Use the wrapped model's EMA version
    else:
        # Load standard model weights
        model.load_state_dict(ckpt["model"])
        model.eval()  # Set to evaluation mode
        print("Loaded non-EMA weights from checkpoint.")
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

    mixture_data = {}
    for t in checkpoints:
        alpha_t = model.alphas_cumprod[t]
        sqrt_alpha = torch.sqrt(alpha_t)
        var_t = (1.0 - alpha_t).item()
        mus = sqrt_alpha * all_x0  # shape [N,1,28,28]
        mixture_data[t] = (mus, var_t)

    return mixture_data

# ============================
# Scoring Function
# ============================
def score_candidates(approach, distribution_data, t, candidates):
    """
    Compute the score for each candidate at time step t,
    given the distribution_data and chosen 'approach'.
    Return a 1D tensor of scores (higher is better).
    """
    if approach == "mse":
        if t not in distribution_data:
            return None
        target_t = distribution_data[t]  # shape [1,1,28,28]
        target = target_t.expand(candidates.shape[0], -1, -1, -1)
        # Negative MSE
        scores = -F.mse_loss(candidates, target, reduction='none').mean(dim=[1,2,3])
        return scores

    elif approach == "bayes":
        posterior_means, posterior_vars = distribution_data
        if t not in posterior_means:
            return None
        mu_t = posterior_means[t].expand(candidates.shape[0], -1, -1, -1)
        var_t = posterior_vars[t]  # scalar float
        # Negative squared error / (2 * var_t)
        squared_errors = F.mse_loss(candidates, mu_t, reduction='none').mean(dim=[1,2,3])
        scores = -squared_errors / (2.0 * var_t)
        return scores

    elif approach == "mixture":
        if t not in distribution_data:
            return None
        mus, var_t = distribution_data[t]  # mus: [N,1,28,28], var_t: float
        batch_size = candidates.shape[0]
        c_expanded = candidates.unsqueeze(1)  # [batch,1,1,28,28]
        m_expanded = mus.unsqueeze(0)         # [1,N,1,28,28]
        diff = c_expanded - m_expanded
        # Mean across spatial dims
        sq_dist = diff.pow(2).mean(dim=[2,3,4])  # [batch, N]
        exponent = -sq_dist / (2.0 * var_t)
        # Log-sum-exp trick across the mixture dimension
        max_exponent, _ = exponent.max(dim=1, keepdim=True)
        exponent_shifted = exponent - max_exponent
        sum_exp = torch.exp(exponent_shifted).sum(dim=1)
        mixture_sum = (1.0 / mus.shape[0]) * torch.exp(max_exponent.squeeze()) * sum_exp
        ll = torch.log(mixture_sum + 1e-12)
        return ll

    else:
        raise ValueError(f"Unknown approach: {approach}")

# ============================
# Search Method #1 (Top-K)
# ============================
def top_k_search(model, distribution_data, approach, n_candidates=64, device="cuda"):
    """
    'Top-K' search (formerly known as reverse_diffusion_search).
    Uses the global CHECKPOINTS to prune, step by step, from T-1 down to 0.
    """
    candidates = torch.randn(
        (n_candidates, model.in_channels, model.image_size, model.image_size),
        device=device
    )

    for t in range(model.timesteps - 1, -1, -1):
        t_tensor = torch.full((candidates.shape[0],), t, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)
        candidates = model._reverse_diffusion(candidates, t_tensor, noise)

        # Prune if we have distribution data for this t
        scores_t = score_candidates(approach, distribution_data, t, candidates)
        if scores_t is not None:
            k = max(1, candidates.shape[0] // 2)
            topk_indices = torch.topk(scores_t, k=k).indices
            candidates = candidates[topk_indices]

    # Final pick from the last batch
    scores_final = score_candidates(approach, distribution_data, 0, candidates)
    if scores_final is not None:
        best_idx = torch.argmax(scores_final)
        best_candidate = candidates[best_idx]
    else:
        best_candidate = candidates[0]

    return best_candidate

# ============================
# Search Method #2 (Paths)
# ============================
def _partial_forward_diffusion(model, x_t, t_from, t_to, noise):
    """
    Forward-diffuse from time 't_from' to time 't_to' (t_to > t_from).
    For an unconditional model, the formula is:
       x_tTo = sqrt( α_cum[t_to]/α_cum[t_from] ) * x_t +
               sqrt(1 - α_cum[t_to]/α_cum[t_from]) * noise
    """
    if t_to <= t_from:
        return x_t  # No forward diffusion needed

    alpha_ratio = model.alphas_cumprod[t_to] / model.alphas_cumprod[t_from]
    alpha_ratio = alpha_ratio.sqrt()  # scale for the mean
    var_ratio = (1.0 - model.alphas_cumprod[t_to]/model.alphas_cumprod[t_from]).sqrt()
    return alpha_ratio * x_t + var_ratio * noise


def get_path_checkpoints(num_steps, delta_f, delta_b):
    """
    For "search_over_paths", define a sequence of partial forward/back steps
    by chunking timesteps with intervals delta_f, delta_b.

    Returns a list of 'reverse checkpoint' times (descending).
    E.g. [900, 840, 780, ... , 0]
    """
    checkpoints = []
    cur_step = num_steps - 1
    while (cur_step - delta_b) >= 0:
        cur_step -= delta_b
        checkpoints.append(cur_step)
        cur_step += delta_f
    # Ensure time=0 is included
    if 0 not in checkpoints:
        checkpoints.append(0)
    checkpoints.sort(reverse=True)
    return checkpoints


def search_over_paths(
    model,
    distribution_data,
    approach,
    n_candidates=64,
    device="cuda",
    delta_f=30,
    delta_b=60
):
    """
    The "paths" search:
      1) Start with n_candidates random x_{T}
      2) Reverse-diffuse delta_b steps => x_{T - delta_b}, then prune to top K
      3) Duplicate each top candidate, forward diffuse delta_f steps => x_{T - delta_b + delta_f}
      4) Repeat until we eventually reach t=0
      5) Return best candidate
    """
    t_current = model.timesteps - 1
    candidates = torch.randn(
        (n_candidates, model.in_channels, model.image_size, model.image_size),
        device=device
    )

    rev_checkpoints = get_path_checkpoints(model.timesteps, delta_f, delta_b)

    for ckpt_t in rev_checkpoints:
        # Reverse diffuse from t_current down to ckpt_t
        while t_current > ckpt_t:
            t_tensor = torch.full((candidates.shape[0],), t_current, device=device, dtype=torch.long)
            noise = torch.randn_like(candidates)
            candidates = model._reverse_diffusion(candidates, t_tensor, noise)
            t_current -= 1

        # Score & prune (if we have data for ckpt_t)
        scores_ckpt = score_candidates(approach, distribution_data, ckpt_t, candidates)
        if scores_ckpt is not None:
            k = max(1, candidates.shape[0] // 2)
            topk_indices = torch.topk(scores_ckpt, k=k).indices
            candidates = candidates[topk_indices]

        # If we've reached t=0, no forward step needed
        if ckpt_t == 0:
            break

        # Forward diffuse from ckpt_t to ckpt_t + delta_f
        next_ckpt = ckpt_t + delta_f
        candidates = candidates.repeat_interleave(n_candidates, dim=0)

        noise = torch.randn_like(candidates)
        candidates = _partial_forward_diffusion(
            model=model,
            x_t=candidates,
            t_from=ckpt_t,
            t_to=next_ckpt,
            noise=noise
        )

        t_current = next_ckpt

    # Final pick from the last batch
    scores_final = score_candidates(approach, distribution_data, 0, candidates)
    if scores_final is not None:
        best_idx = torch.argmax(scores_final)
        best_candidate = candidates[best_idx]
    else:
        best_candidate = candidates[0]

    return best_candidate

# ============================
# New Helper: pick distribution checkpoints
# ============================

def get_distribution_checkpoints_for_search(model, search_method):
    """
    Return a list of timesteps for which we will compute distribution data.
    - If search_method='top_k', we use the global CHECKPOINTS
    - If search_method='paths', we build a list from get_path_checkpoints()
      so we can prune at all relevant path steps.
    """
    if search_method == "top_k":
        return CHECKPOINTS
    elif search_method == "paths":
        path_ckpts = get_path_checkpoints(model.timesteps, DELTA_F, DELTA_B)
        # We also always want t=0 in the distribution set,
        # but get_path_checkpoints already ensures 0 is included.
        # Optionally, ensure T-1 is in there for completeness:
        # but we only need distribution at the reverse checkpoints themselves + t=0
        return path_ckpts
    else:
        raise ValueError(f"Unknown search method: {search_method}")

# ============================
# Search Dispatcher
# ============================
def perform_search(model, distribution_data, approach, device, search_method_name="top_k"):
    """
    Calls either 'top_k_search' or 'search_over_paths' with the appropriate n_candidates.
    """
    if search_method_name == "top_k":
        return top_k_search(
            model=model,
            distribution_data=distribution_data,
            approach=approach,
            n_candidates=N_CANDIDATES_TOP_K,
            device=device
        )
    elif search_method_name == "paths":
        return search_over_paths(
            model=model,
            distribution_data=distribution_data,
            approach=approach,
            n_candidates=N_CANDIDATES_PATHS,
            device=device,
            delta_f=DELTA_F,
            delta_b=DELTA_B
        )
    else:
        raise ValueError(f"Unknown search method: {search_method_name}")

# ============================
# Distribution Dispatcher
# ============================
def get_distribution_for_digit(model, digit_loader, digit, checkpoints, device, approach):
    """Compute distribution info for the given approach at each time in 'checkpoints'."""
    if approach == "mse":
        return estimate_target_distribution_mse(model, digit_loader, digit, checkpoints, device)
    elif approach == "bayes":
        return estimate_target_distribution_bayes(model, digit_loader, digit, checkpoints, device)
    elif approach == "mixture":
        return estimate_target_distribution_mixture(model, digit_loader, digit, checkpoints, device)
    else:
        raise ValueError(f"Unknown approach: {approach}")

# ============================
# Main Generation Helper
# ============================
def generate_samples_for_digit(
    model,
    digit,
    verifier_data_subset_size,
    approach="mse",
    search_method="top_k",
    n_experiments=10,
    device="cuda"
):
    """
    1) Load data for that digit.
    2) Determine the relevant distribution checkpoints (depends on search_method).
    3) Compute or fetch distribution for 'approach' at those checkpoints.
    4) Perform generation n_experiments times using 'search_method'.
    5) Return list of (image_tensor, intended_digit).
    """
    # Choose which time steps to compute distribution for,
    # depending on whether we are top_k or paths.
    distribution_checkpoints = get_distribution_checkpoints_for_search(model, search_method)

    # Dist cache key now includes search_method
    dist_key = (digit, verifier_data_subset_size, approach, search_method)
    if dist_key not in _distribution_cache:
        print(f"  -> Computing distribution for digit={digit}, subset_size={verifier_data_subset_size}, "
              f"approach={approach}, search_method={search_method}")
        digit_loader = create_digit_dataloader(digit=digit, subset_size=verifier_data_subset_size)
        dist_data = get_distribution_for_digit(
            model=model,
            digit_loader=digit_loader,
            digit=digit,
            checkpoints=distribution_checkpoints,
            device=device,
            approach=approach
        )
        _distribution_cache[dist_key] = dist_data
    else:
        dist_data = _distribution_cache[dist_key]
        print(f"  -> Using cached distribution for digit={digit}, subset_size={verifier_data_subset_size}, "
              f"approach={approach}, search_method={search_method}")

    out_samples = []
    for exp_idx in range(n_experiments):
        print(f"    [Digit {digit}] Generating sample {exp_idx+1}/{n_experiments} "
              f"with approach={approach}, search={search_method}")
        best_noise = perform_search(
            model=model,
            distribution_data=dist_data,
            approach=approach,
            device=device,
            search_method_name=search_method
        )
        out_samples.append((best_noise.unsqueeze(0), digit))

    return out_samples

# ============================
# Scaling Study
# ============================
def run_scaling_study(
    diffusion_model,
    hf_model,
    feature_extractor,
    verifier_data_sizes=[10, 50, 100],
    approach="mse",
    search_method="top_k",
    n_experiments_per_digit=3,
    device="cuda"
):
    """
    Evaluate classification accuracy for different subset sizes.
    Returns { subset_size : accuracy }.
    """
    results = {}

    for subset_size in verifier_data_sizes:
        print(f"\n=== Running experiments for subset_size={subset_size}, approach={approach}, "
              f"search={search_method} ===")

        generated_samples = []
        for digit in range(10):
            digit_samples = generate_samples_for_digit(
                model=diffusion_model,
                digit=digit,
                verifier_data_subset_size=subset_size,
                approach=approach,
                search_method=search_method,
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

        print(f"  => Approach={approach.upper()} | Search={search_method} | "
              f"subset_size={subset_size} | Accuracy={accuracy*100:.2f}% ({correct}/{total})")

        results[subset_size] = accuracy

        # Optionally: confusion matrix
        cm = confusion_matrix(all_intended_digits, predicted_labels, labels=list(range(10)))
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted")
        plt.ylabel("Intended")
        plt.title(f"CM (Subset={subset_size}, Approach={approach}, Search={search_method})")
        cm_path = os.path.join(LOG_DIR, f"cm_{approach}_{search_method}_subset_{subset_size}.png")
        plt.savefig(cm_path, dpi=150)
        plt.close()

    return results

def plot_scaling_study_results(results_dict, approach="mse", search_method="top_k"):
    """
    Plot accuracy vs. subset_size from results_dict = {subset_size: accuracy}
    """
    sizes = sorted(results_dict.keys())
    accuracies = [results_dict[s] for s in sizes]

    plt.figure()
    plt.plot(sizes, accuracies, marker='o')
    plt.xlabel("Number of Labeled Examples (per digit) for Verifier")
    plt.ylabel("Classification Accuracy")
    plt.title(f"Approach={approach.upper()}, Search={search_method}: Accuracy vs. Verifier Data Size")
    plot_path = os.path.join(LOG_DIR, f"scaling_study_accuracy_{approach}_{search_method}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved scaling study plot to {plot_path}")

# ============================
# Main Experiment
# ============================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(TRAINED_MODELS_DIR, CHECKPOINT)

    print(f"Loading diffusion model from: {ckpt_path}")
    diffusion_model = load_diffusion_model(
        ckpt_path,
        device=device,
        use_ema=USE_EMA
    )

    # Load Hugging Face classifier
    hf_model, feature_extractor = load_hf_classifier(device=device)

    # For logging
    full_log_path = os.path.join(LOG_DIR, "combined_experiment_log.txt")
    with open(full_log_path, "w") as f_log:
        # Write high-level config
        f_log.write("=== EXPERIMENT DETAILS ===\n")
        f_log.write(f"Timestamp: {timestamp}\n")
        f_log.write(f"MODEL_TYPE: {MODEL_TYPE}\n")
        f_log.write(f"USE_EMA: {USE_EMA}\n")
        f_log.write(f"Checkpoints (used only for top_k): {CHECKPOINTS}\n\n")

        # Record candidate settings
        f_log.write(f"N_CANDIDATES_TOP_K: {N_CANDIDATES_TOP_K}\n")
        f_log.write(f"N_CANDIDATES_PATHS: {N_CANDIDATES_PATHS}\n")

        f_log.write(f"DELTA_F: {DELTA_F}, DELTA_B: {DELTA_B}\n")
        f_log.write(f"verifier_data_sizes: {VERIFIER_DATA_SIZES}\n")
        f_log.write(f"n_experiments_per_digit: {N_EXPERIMENTS_PER_DIGIT}\n")
        f_log.write(f"Diffusion Model Checkpoint: {CHECKPOINT}\n")
        f_log.write(f"HF_MODEL_NAME: {HF_MODEL_NAME}\n\n")

        # Nested loops over approach & search method
        for approach in APPROACHES_TO_TRY:
            for search_method in SEARCH_METHODS_TO_TRY:
                print(f"\n======== Starting scaling study for approach={approach}, search={search_method} ========\n")
                f_log.write(f"=== Approach={approach}, Search={search_method} ===\n")

                # Run scaling study
                results = run_scaling_study(
                    diffusion_model=diffusion_model,
                    hf_model=hf_model,
                    feature_extractor=feature_extractor,
                    verifier_data_sizes=VERIFIER_DATA_SIZES,
                    approach=approach,
                    search_method=search_method,
                    n_experiments_per_digit=N_EXPERIMENTS_PER_DIGIT,
                    device=device
                )

                # Plot results
                plot_scaling_study_results(results, approach=approach, search_method=search_method)

                # Save separate text summary
                approach_result_path = os.path.join(LOG_DIR, f"results_{approach}_{search_method}.txt")
                with open(approach_result_path, "w") as f_approach:
                    f_approach.write(f"Approach={approach}, Search={search_method}\n")
                    f_approach.write(f"(For top_k, used global CHECKPOINTS: {CHECKPOINTS})\n")
                    f_approach.write(f"N_CANDIDATES_TOP_K: {N_CANDIDATES_TOP_K}\n")
                    f_approach.write(f"N_CANDIDATES_PATHS: {N_CANDIDATES_PATHS}\n")
                    f_approach.write(f"DELTA_F={DELTA_F}, DELTA_B={DELTA_B}\n")
                    f_approach.write(f"verifier_data_sizes: {VERIFIER_DATA_SIZES}\n")
                    f_approach.write(f"n_experiments_per_digit: {N_EXPERIMENTS_PER_DIGIT}\n")
                    f_approach.write("Per-subset accuracy results:\n")
                    for sz, acc in results.items():
                        f_approach.write(f"  subset_size={sz}, accuracy={acc:.4f}\n")

                # Also append to combined log file
                f_log.write(f"Approach={approach}, Search={search_method}\n")
                for sz, acc in results.items():
                    f_log.write(f"  subset_size={sz}, accuracy={acc:.4f}\n")
                f_log.write("\n")

    print(f"\nAll experiments complete. Detailed logs saved to {full_log_path}")
    print("Done!")
    print(f"Note: For 'paths' search, distribution info was computed for timesteps={get_path_checkpoints(diffusion_model.timesteps, DELTA_F, DELTA_B)}")

if __name__ == "__main__":
    main()
