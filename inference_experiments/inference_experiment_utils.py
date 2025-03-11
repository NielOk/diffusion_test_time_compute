import sys
import os
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.linalg import sqrtm  # For FID
from collections import defaultdict
import torch.profiler as profiler

# Transformers / HF
from transformers import AutoModelForImageClassification, AutoFeatureExtractor

HF_MODEL_NAME = "farleyknight/mnist-digit-classification-2022-09-04"

INFERENCE_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(INFERENCE_EXPERIMENTS_DIR)
MNIST_ROOT = "./mnist_data"

# ============================
# Code and model loading
# ============================

def load_code(model_type='lc'):
    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')
    
    if 'model' in sys.modules:
        del sys.modules['model']
    if 'utils' in sys.modules:
        del sys.modules['utils']
    if 'train_mnist' in sys.modules:
        del sys.modules['train_mnist']
    if 'unet' in sys.modules:
        del sys.modules['unet']

    # Get directories
    lc_gpu_accelerated_training = os.path.join(REPO_DIR, 'lc_gpu_accelerated_training')
    LC_TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'lc_trained_ddpm', 'results')
    nlc_gpu_accelerated_training = os.path.join(REPO_DIR, 'nlc_gpu_accelerated_training')
    NLC_TRAINED_MODELS_DIR = os.path.join(REPO_DIR, 'nlc_trained_ddpm', 'results')

    if model_type == 'lc':
        sys.path.append(lc_gpu_accelerated_training)

        if nlc_gpu_accelerated_training in sys.path:
            sys.path.remove(nlc_gpu_accelerated_training)
        
        from train_mnist import create_mnist_dataloaders
        from model import MNISTDiffusion
        from utils import ExponentialMovingAverage

        return LC_TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage

    elif model_type == 'nlc':
        sys.path.append(nlc_gpu_accelerated_training)
        if lc_gpu_accelerated_training in sys.path:
            sys.path.remove(lc_gpu_accelerated_training)

        from train_mnist import create_mnist_dataloaders
        from model import MNISTDiffusion
        from utils import ExponentialMovingAverage

        return NLC_TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage
    
# -----------------------------------------------------------------------------
# 1) FID SUPPORT: MNIST Inception + feature extraction + FID calculation
# -----------------------------------------------------------------------------

class MnistInceptionModel(torch.nn.Module):
    """
    Minimal "Inception-like" model for MNIST, purely for demonstration.
    Outputs a 128-D penultimate layer for FID calculation.
    """
    def __init__(self):
        super().__init__()
        self.features_layer = torch.nn.Sequential(
            torch.nn.Linear(28*28, 128),  # expects 1×28×28 => 784
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Linear(128, 10)  # 10 classes for MNIST

    def forward(self, x):
        """
        x shape: [batch, 1, 28, 28].
        We'll flatten, feed to features, then classify.
        """
        b, c, h, w = x.shape
        x = x.view(b, -1)          # shape [b, 784]
        feats = self.features_layer(x)  # shape [b, 128]
        logits = self.classifier(feats) # shape [b, 10]
        return logits, feats

    def features(self, x):
        """
        Returns the penultimate-layer (128-D) embeddings.
        """
        _, feats = self.forward(x)
        return feats

def get_mnist_features(images, fid_model, device):
    """
    images : list of PIL Images
    returns: np.array shape [N,128] of penultimate-layer features
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    feats_list = []
    for img in images:
        img_t = transform(img).unsqueeze(0).to(device)  # shape [1,1,28,28]
        with torch.no_grad():
            feats = fid_model.features(img_t)  # shape [1,128]
        feats_list.append(feats.squeeze(0).cpu().numpy())
    return np.vstack(feats_list)

def compute_fid(x_feats, y_feats):
    """
    Frechet Inception Distance (FID) between two sets of features.
    x_feats: [N,128]
    y_feats: [M,128]
    """
    mu_x = np.mean(x_feats, axis=0)
    mu_y = np.mean(y_feats, axis=0)
    sigma_x = np.cov(x_feats, rowvar=False)
    sigma_y = np.cov(y_feats, rowvar=False)

    diff = mu_x - mu_y
    diff_sq = diff.dot(diff)

    cov_prod = sigma_x.dot(sigma_y)
    cov_sqrt, _ = sqrtm(cov_prod, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid_val = diff_sq + np.trace(sigma_x + sigma_y - 2 * cov_sqrt)
    return fid_val

#============
# 2) Helpers
#============
    
def create_digit_dataloader(digit, subset_size=None, batch_size=128, image_size=28, num_workers=0, debug=False):
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

    if debug and len(indices) > 0:
        sample_image, sample_label = full_dataset[indices[0]]
        plt.imshow(sample_image.squeeze(), cmap='gray')
        plt.title(f"Sample Digit: {sample_label}")
        plt.show()

    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_model_paths(TRAINED_MODELS_DIR):
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

def load_model_architecture(MNISTDiffusion,device='cpu', model_type='lc'):
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

# ============================
# Hugging Face MNIST classifier
# ============================

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


# ==================
# Scoring Function
# ==================

def score_candidates(approach, distribution_data, t, candidates):
    """
    Compute the score for each candidate at time step t,
    by using the distribution data and the specified approach.
    Returns a 1D tensor of scores (higher is better).
    """
    B, K, C, H, W = candidates.shape

    if approach == "mse":
        if t not in distribution_data:
            return None
        target_t = distribution_data[t]  # shape [1,1,28,28]
        target = target_t.expand(B, K, -1, -1, -1).reshape(B * K, C, H, W)  # 🔹 Correct expansion
        scores = -F.mse_loss(candidates.view(B * K, C, H, W), target, reduction='none').mean(dim=[1,2,3])
        return scores.view(B, K)  # 🔹 Reshape scores back to (B, K)

    elif approach == "bayes":
        posterior_means, posterior_vars = distribution_data
        if t not in posterior_means:
            return None
        mu_t = posterior_means[t].expand(B, K, -1, -1, -1).reshape(B * K, C, H, W)  # 🔹 Correct expansion
        var_t = posterior_vars[t]  # scalar float
        squared_errors = F.mse_loss(candidates.view(B * K, C, H, W), mu_t, reduction='none').mean(dim=[1,2,3])
        scores = -squared_errors / (2.0 * var_t)
        return scores.view(B, K)  # 🔹 Reshape back to (B, K)

    elif approach == "mixture":
        if t not in distribution_data:
            return None
        mus, var_t = distribution_data[t]  # mus: [N,1,28,28], var_t: float
        c_expanded = candidates.view(B, K, C, H, W).unsqueeze(2)  # 🔹 (B, K, 1, C, H, W)
        m_expanded = mus.unsqueeze(0).unsqueeze(0)  # 🔹 (1, 1, N, C, H, W)
        diff = c_expanded - m_expanded
        sq_dist = diff.pow(2).mean(dim=[3,4,5])  # 🔹 (B, K, N)
        exponent = -sq_dist / (2.0 * var_t)
        max_exponent, _ = exponent.max(dim=2, keepdim=True)
        exponent_shifted = exponent - max_exponent
        sum_exp = torch.exp(exponent_shifted).sum(dim=2)
        mixture_sum = (1.0 / mus.shape[0]) * torch.exp(max_exponent.squeeze()) * sum_exp
        ll = torch.log(mixture_sum + 1e-12)
        return ll.view(B, K)  # 🔹 Reshape back to (B, K)

    else:
        raise ValueError(f"Unknown approach: {approach}")
    

# ============================
# Search method #1: (Top-k))
# ============================
def top_k_search(model, model_ema, distribution_data, approach, digit_to_generate, batch_size=16, ema=True, use_clip=True, model_type="lc", n_candidates=64, device="cuda"):
    """
    'Top-K' search (formerly known as reverse_diffusion_search).
    Uses the global CHECKPOINTS to prune, step by step, from T-1 down to 0.
    """
    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')
    
    candidates = torch.randn((batch_size, n_candidates, model.in_channels, model.image_size, model.image_size), device=device)

    for t in range(model.timesteps - 1, -1, -1):
        B, K, C, H, W = candidates.shape
        t_tensor = torch.full((B * K,), t, device=device, dtype=torch.long)
        candidates = candidates.view(B * K, C, H, W)
        noise = torch.randn_like(candidates)
        labels = torch.full((B * K,), digit_to_generate, device=device, dtype=torch.long)
        if model_type == 'lc':
            if ema:
                if use_clip:
                    candidates = model_ema.module._reverse_diffusion_with_clip(candidates, t_tensor, noise, labels)
                else:
                    candidates = model_ema.module._reverse_diffusion(candidates, t_tensor, noise, labels)
            else:
                if use_clip:
                    candidates = model._reverse_diffusion_with_clip(candidates, t_tensor, noise, labels)
                else:
                    candidates = model._reverse_diffusion(candidates, t_tensor, noise, labels)
        else:
            if ema:
                if use_clip:
                    candidates = model_ema.module._reverse_diffusion_with_clip(candidates, t_tensor, noise)
                else:
                    candidates = model_ema.module._reverse_diffusion(candidates, t_tensor, noise)
            else:
                if use_clip:
                    candidates = model._reverse_diffusion_with_clip(candidates, t_tensor, noise)
                else:
                    candidates = model._reverse_diffusion(candidates, t_tensor, noise)

        candidates = candidates.view(B, K, C, H, W)

        # Prune if we have distribution data for this t
        scores_t = score_candidates(approach, distribution_data, t, candidates)
        if scores_t is not None:
            k = max(1, K // 2)
            topk_indices = torch.topk(scores_t, k=k, dim=1).indices  # Shape: (B, k)
            # Gather top-k candidates per batch
            candidates = torch.gather(
                candidates, dim=1,
                index=topk_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, H, W)
            )
            K = k  # Update K dynamically after pruning

    candidates = candidates.view(B, K, C, H, W)
    # Final pick from the last batch
    scores_final = score_candidates(approach, distribution_data, 0, candidates)
    if scores_final is not None:
        best_indices = torch.argmax(scores_final, dim=1)
        
        # Select the best candidate per batch
        best_candidates = candidates[torch.arange(B, device=device), best_indices]
    else:
        # Select the first candidate for each batch
        best_candidates = candidates[:, 0]

    return best_candidates


# ======================================
# Search method #2: (Search-over-paths)
# ======================================
def denoise_to_step(model, candidates, t, start_point, labels, B, K, model_type="lc", device="cpu", ema=False, use_clip=True):

    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')
            
    for step in range(start_point, t, -1):
        t_tensor = torch.full((B * K,), step, device=device, dtype=torch.long)
        noise = torch.randn_like(candidates)

        if model_type == 'lc':
            if ema:
                if use_clip:
                    candidates = model.module._reverse_diffusion_with_clip(candidates, t_tensor, noise, labels)
                else:
                    candidates = model.module._reverse_diffusion(candidates, t_tensor, noise, labels)
            else:
                if use_clip:
                    candidates = model._reverse_diffusion_with_clip(candidates, t_tensor, noise, labels)
                else:
                    candidates = model._reverse_diffusion(candidates, t_tensor, noise, labels)
        else:
            if ema:
                if use_clip:
                    candidates = model.module._reverse_diffusion_with_clip(candidates, t_tensor, noise)
                else:
                    candidates = model.module._reverse_diffusion(candidates, t_tensor, noise)
            else:
                if use_clip:
                    candidates = model._reverse_diffusion_with_clip(candidates, t_tensor, noise)
                else:
                    candidates = model._reverse_diffusion(candidates, t_tensor, noise)

    return candidates  # Denoised images at step `t`

def get_checkpoints(delta_f, delta_b, num_steps=1000):

    checkpoints = []
    cur_step = num_steps - 1
    while cur_step - delta_b >= 0:
        cur_step -= delta_b
        checkpoints.append(cur_step)
        cur_step += delta_f

    checkpoints.append(0) # Step 0 is always a checkpoint

    print("Checkpoints:", checkpoints) # Checkpoints are where we will evaluate the candidates, or have scoring methods for the candidates.

    return checkpoints

def search_over_paths(n_candidates, delta_f, delta_b, model, model_ema, digit_to_generate, distribution_data, batch_size=16, model_type="lc", ema=False, use_clip=True, device='cpu', scoring_approach='mse'):
    if delta_f > delta_b:
        raise ValueError("delta_f must be less than delta_b.")
    
    if model_type not in ['lc', 'nlc']:
        raise ValueError('model_type must be one of "lc" or "nlc"')
    
    if scoring_approach not in ['mse', 'bayes', 'mixture']:
        raise ValueError('scoring_approach must be one of "mse", "bayes", or "mixture"')
    
    # Get first set of candidates
    candidates = torch.randn((batch_size, n_candidates, model.in_channels, model.image_size, model.image_size), device=device)

    # Right up until the last checkpoint
    t = model.timesteps - 1
    while (t - delta_b) >= 0:

        B, K, C, H, W = candidates.shape
        candidates = candidates.view(B * K, C, H, W)

        noise = torch.randn_like(candidates, device=device)
        labels = torch.full((B * K,), digit_to_generate, device=device, dtype=torch.long)

        # Denoise candidates to checkpoint
        if ema:
            candidates = denoise_to_step(model_ema, candidates, t - delta_b, t, labels, B, K, model_type=model_type, device=device, ema=ema, use_clip=use_clip)
        else:
            candidates = denoise_to_step(model, candidates, t - delta_b, t, labels, B, K, model_type=model_type, device=device, ema=ema, use_clip=use_clip)

        candidates = candidates.view(B, K, C, H, W)

        t -= delta_b
        # Select top n candidates at checkpoint with scoring method
        scores_ckpt = score_candidates(scoring_approach, distribution_data, t, candidates)
        k = min(K, n_candidates)
        topk_indices = torch.topk(scores_ckpt, k=k, dim=1).indices  # Shape: (B, k)
        # Gather top-k candidates per batch
        candidates = torch.gather(
            candidates, dim=1,
            index=topk_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, H, W)
        )
        K = k # Update K dynamically after pruning

        print(f'Finished timestep {t}, kept {K} candidates per batch')

        # Expand each top candidate to n copies before renoising
        candidates = candidates.repeat_interleave(n_candidates, dim=1)  # Expands from n to n*10
        K = K * n_candidates  # Update K to reflect the new candidate count

        candidates = candidates.view(B * K, C, H, W)
        noise = torch.randn_like(candidates, device=device)  # Generate fresh noise for each expanded candidate

        # Renoise candidates
        candidates = model._partial_forward_diffusion(
            candidates,
            torch.full((B * K,), t, device=device, dtype=torch.long),
            torch.full((B * K,), t + delta_f, device=device, dtype=torch.long),
            noise
        )
        candidates = candidates.view(B, K, C, H, W)

        t += delta_f

    candidates = candidates.view(B * K, C, H, W)
    # Last checkpoint, denoise candidate to step 0
    noise = torch.randn_like(candidates, device=device)
    labels = torch.full((B * K,), digit_to_generate, device=device, dtype=torch.long)
    if ema:
        candidates = denoise_to_step(model_ema, candidates, 0, t, labels, B, K, model_type=model_type, device=device, ema=ema, use_clip=use_clip)
    else:
        candidates = denoise_to_step(model, candidates, 0, t, labels, B, K, model_type=model_type, device=device, ema=ema, use_clip=use_clip)

    # Select best candidate
    candidates = candidates.view(B, K, C, H, W)
    scores_0 = score_candidates(scoring_approach, distribution_data, 0, candidates)
    best_indices = torch.argmax(scores_0, dim=1)
    best_candidates = candidates[torch.arange(B, device=device), best_indices]

    return best_candidates

# ============================
# Search Dispatcher
# ============================
def perform_search(model, model_ema, distribution_data, approach, device, n_candidates, search_method_name="top_k", model_type="lc", batch_size=16, delta_f=None, delta_b=None, digit_to_generate=None, ema=True, use_clip=True):
    """
    Calls either 'top_k_search' or 'search_over_paths', passing the dist_key
    so that scoring can fetch distribution data from `_distribution_cache`.
    """

    print(model_type)

    if search_method_name == "top_k":
        return top_k_search(
            model, 
            model_ema, 
            distribution_data, 
            approach, 
            digit_to_generate, 
            batch_size=batch_size, 
            ema=ema, 
            use_clip=use_clip, 
            model_type=model_type, 
            n_candidates=n_candidates, 
            device=device)
    elif search_method_name == "paths":
        return search_over_paths(
            n_candidates, 
            delta_f, 
            delta_b,
            model, 
            model_ema, 
            digit_to_generate, 
            distribution_data, 
            batch_size=batch_size, 
            model_type=model_type, 
            ema=ema, 
            use_clip=use_clip, 
            device=device, 
            scoring_approach=approach)
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
        model_ema,
        digit_to_generate,
        verifier_data_subset_size, 
        n_candidates=5,
        delta_f=10,
        delta_b=30,
        num_steps=1000,
        model_type="lc",
        approach="mse",
        search_method="top_k",
        n_experiments=10,
        device="cuda",
        ema=True,
        use_clip=True
):
    
    checkpoints = get_checkpoints(delta_f, delta_b, num_steps=num_steps)
    
    digit_loader = create_digit_dataloader(digit=digit_to_generate, subset_size=verifier_data_subset_size, batch_size=128)
    distribution_data = get_distribution_for_digit(model, digit_loader, digit_to_generate, checkpoints, device, approach)

    out_samples = []
    if search_method == "top_k":
        batch_size = min(n_experiments, 50)  # Generate up to 16 images at a time
    if search_method == "paths":
        batch_size = min(n_experiments, 10)  # Generate up to 8 images at a time

    num_batches = n_experiments // batch_size
    for batch_idx in range(num_batches):
        if batch_idx == num_batches - 1:
            print(f"   [Digit {digit_to_generate}] Generating {n_experiments - len(out_samples)} samples through sample number {n_experiments} "
                  f"with approach={approach}, search={search_method}")
        else:
            print(f"    [Digit {digit_to_generate}] Generating {batch_size} samples through sample number {(batch_idx + 1)*batch_size} "
              f"with approach={approach}, search={search_method}")
            
        batch_noise = perform_search(
            model=model,
            model_ema=model_ema,
            distribution_data=distribution_data,
            approach=approach,
            device=device,
            n_candidates=n_candidates,
            search_method_name=search_method,
            model_type=model_type,
            batch_size=batch_size,
            delta_f=delta_f,
            delta_b=delta_b,
            digit_to_generate=digit_to_generate,
            ema=ema,
            use_clip=use_clip
        )

        for img in batch_noise:
            out_samples.append((img.unsqueeze(0), digit_to_generate))

    return out_samples

# -----------------------------------------------------------------------------
# 2) FID HELPER: Gather real test-set images + compute FID vs. generated
# -----------------------------------------------------------------------------

def get_test_pil_images_for_digit(digit, num_images_needed):
    """
    Gather `num_images_needed` PIL images of the given digit from *test* set (not train).
    """
    mnist_test = datasets.MNIST(
        root=MNIST_ROOT,
        train=False,
        download=True,
        transform=None
    )
    images = []
    for img, label in mnist_test:
        if label == digit:
            images.append(img)  # raw PIL
            if len(images) == num_images_needed:
                break
    return images

def compute_fid_for_generated_samples(generated_samples, fid_model, device="cuda"):
    """
    Given a list of (torch_image, digit) pairs in `generated_samples`,
    gather an equal number of real test-set images per digit, extract features,
    and compute the FID (one big real-vs-generated set).
    """
    gen_by_digit = defaultdict(list)

    # Convert each generated sample to a PIL image and group by digit
    for (torch_img, d) in generated_samples:
        arr = torch_img.squeeze(0).detach().cpu().numpy().squeeze()
        arr_255 = (arr + 1.0) / 2.0 * 255.0
        arr_255 = np.clip(arr_255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr_255, mode='L')
        gen_by_digit[d].append(pil_img)

    all_generated_pil = []
    all_real_pil = []

    # For each digit, gather same count of real test images
    for d in range(10):
        gen_list = gen_by_digit[d]
        num_gen = len(gen_list)
        real_list = get_test_pil_images_for_digit(d, num_gen)
        all_generated_pil.extend(gen_list)
        all_real_pil.extend(real_list)

    # Extract features
    real_feats = get_mnist_features(all_real_pil, fid_model, device)
    gen_feats  = get_mnist_features(all_generated_pil, fid_model, device)

    # Compute FID
    fid_val = compute_fid(real_feats, gen_feats)
    return fid_val

# --------------
# FLOP COUNTING
# --------------

def get_operation_flop_dict(
        model,
        model_ema, 
        digit_to_generate,
        batch_size=16,
        n_candidates=5,
        model_type="lc",
        approach="mse",
        device="cuda",
        ema=True,
        use_clip=True
):
    
    candidates = torch.randn((batch_size, n_candidates, model.in_channels, model.image_size, model.image_size), device=device)
    B, K, C, H, W = candidates.shape
    candidates = candidates.view(B * K, C, H, W)
    
    flops_dict = {}


def flop_count_measurement(
    model, 
    model_ema,
    digit_to_generate,
    n_candidates=5,
    delta_f=10,
    delta_b=30,
    num_steps=1000,
    model_type="lc",
    approach="mse",
    search_method="top_k",
    n_experiments=10,
    device="cuda",
    ema=True,
    use_clip=True
):
    
    # Flops is same across epochs so just use 100 epochs model 

    TRAINED_MODELS_DIR, create_mnist_dataloaders, MNISTDiffusion, ExponentialMovingAverage = load_code(model_type=model_type)
    
    # Load model architecture
    model = load_model_architecture(MNISTDiffusion, device=device, model_type=model_type)
    model_ema = ExponentialMovingAverage(model, decay=0.995, device=device)

    # Get model filepaths
    sorted_model_paths, sorted_epoch_numbers = get_model_paths(TRAINED_MODELS_DIR)

    epoch_id = 100
    model_to_load = sorted_model_paths[sorted_epoch_numbers.index(epoch_id)]

    # Load model weights
    checkpoint = torch.load(model_to_load, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model_ema.load_state_dict(checkpoint['model_ema'])

    model.eval()
    model_ema.eval()

    if search_method == "top_k":
        batch_size = min(n_experiments, 50)
    if search_method == "paths":
        batch_size = min(n_experiments, 10)

    

    checkpoints = get_checkpoints(delta_f, delta_b, num_steps=num_steps)
