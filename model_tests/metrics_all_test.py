import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm

###############################################################################
# 1. Inception Score utilities
###############################################################################

def load_inception_for_is(device):
    """
    Loads a standard Inception V3 *with* the final classification layer intact.
    We need the model's softmax outputs for Inception Score.
    """
    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    )
    model.eval()
    model.to(device)
    return model

def compute_inception_score(images, inception_model, device, splits=1):
    """
    Computes the Inception Score for a list of PIL images (the 'generated' set).
    Inception Score formula (simplified):
        IS = exp( E_x [ KL( p(y|x) || p(y) ) ] ).

    Steps:
      1) For each image x, get p(y|x) from Inception (softmax).
      2) Average p(y|x) over all images to get p(y).
      3) Compute the KL divergence per image, then average, then exponentiate.

    'splits' is how many times we split the dataset for multiple estimates.
    Usually we do e.g. splits=10 for large sets. With only 10 images, splits=1.
    """
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    
    # Collect softmax probabilities for each image
    probs_list = []
    for img in images:
        img = img.convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = inception_model(img_t)
            probs = torch.nn.functional.softmax(logits, dim=1)
        probs_list.append(probs.cpu().numpy())

    probs_array = np.vstack(probs_list)  # shape [N, 1000]
    N = probs_array.shape[0]
    split_size = N // splits

    is_values = []
    for i in range(splits):
        part = probs_array[i * split_size : i * split_size + split_size]
        
        # Compute p(y) = mean of p(y|x) over all x in this split
        p_y = np.mean(part, axis=0, keepdims=True)
        
        # KL(p(y|x)||p(y)) for each x
        kl_divs = part * (np.log(part + 1e-16) - np.log(p_y + 1e-16))
        kl_mean = np.mean(np.sum(kl_divs, axis=1))
        
        is_values.append(np.exp(kl_mean))

    is_mean = float(np.mean(is_values))
    is_std  = float(np.std(is_values))
    return is_mean, is_std


###############################################################################
# 2. KID and FID utilities (both use Inception "feature extractor")
###############################################################################

def load_inception_for_features(device):
    """
    Loads Inception V3 with the final classification layer replaced by Identity,
    so we can extract 2048-dim pool3 embeddings.
    """
    inception = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    )
    inception.fc = nn.Identity()
    inception.eval()
    inception.to(device)
    return inception

def get_inception_activations(images, inception_model, device):
    """
    Given a list of PIL images, compute the 2048-dim Inception embeddings (pool3).
    Returns a numpy array [N, 2048].
    """
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])
    
    acts_list = []
    for img in images:
        img = img.convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = inception_model(img_t)  # shape [1, 2048]
        acts_list.append(feats.squeeze(0).cpu().numpy())
    
    return np.vstack(acts_list)

def polynomial_kernel(x, y, degree=3, c=1.0):
    """
    Polynomial kernel: K(x,y) = (x^T y / dim + c)^degree.
    x, y: [N, dim] and [M, dim].
    Returns [N, M].
    """
    dot = x @ y.T
    dim = x.shape[1]
    return (dot / dim + c) ** degree

def compute_mmd(x, y, kernel_fn):
    """
    Unbiased MMD^2:
      MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)].
    We'll do the standard unbiased estimator.
    """
    Kxx = kernel_fn(x, x)
    Kyy = kernel_fn(y, y)
    Kxy = kernel_fn(x, y)
    
    n = x.shape[0]
    m = y.shape[0]
    
    # sum_{i!=j} k(x_i, x_j) = sum_{i,j} k(x_i,x_j) - sum_{i} k(x_i, x_i)
    sum_Kxx = np.sum(Kxx) - np.sum(np.diag(Kxx))
    sum_Kyy = np.sum(Kyy) - np.sum(np.diag(Kyy))
    # E[k(x,x')] = sum_{i!=j} k(x_i,x_j)/(n*(n-1))
    mmd_xx = sum_Kxx / (n * (n - 1))
    mmd_yy = sum_Kyy / (m * (m - 1))
    # E[k(x,y)] = sum_{i,j} k(x_i,y_j)/(n*m)
    mmd_xy = 2.0 * np.sum(Kxy) / (n * m)
    
    mmd2 = mmd_xx + mmd_yy - mmd_xy
    return mmd2

def compute_kid(x, y, kernel_fn, splits=1):
    """
    Compute KID by averaging MMD^2 over multiple random splits.
    x, y: [N, 2048], [M, 2048].
    """
    kid_vals = []
    rng = np.random.default_rng(123)  # for reproducibility
    n = x.shape[0]
    m = y.shape[0]
    
    for _ in range(splits):
        # Random shuffle each time
        x_perm = rng.permutation(x)
        y_perm = rng.permutation(y)
        # Optionally sub-sample each split. We'll just use the full set here.
        mmd2 = compute_mmd(x_perm, y_perm, kernel_fn)
        kid_vals.append(mmd2)
    
    return float(np.mean(kid_vals)), float(np.std(kid_vals))

def compute_fid(x, y):
    """
    Compute FID between two sets of embeddings x, y.
    x, y: [N, 2048], [M, 2048].
    """
    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)
    sigma_x = np.cov(x, rowvar=False)
    sigma_y = np.cov(y, rowvar=False)
    
    diff = mu_x - mu_y
    diff_sq = diff.dot(diff)
    
    # Product of covariances
    cov_prod = sigma_x.dot(sigma_y)
    cov_sqrt, _ = sqrtm(cov_prod, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    
    fid_val = diff_sq + np.trace(sigma_x + sigma_y - 2.0 * cov_sqrt)
    return fid_val


###############################################################################
# 3. Main script: Load real/fake images, compute IS, KID, and FID
###############################################################################

def main():
    #---------------------
    # Step A: Load 10 real MNIST '1' images
    #---------------------
    mnist_dataset = torchvision.datasets.MNIST(
        root='./mnist_data',
        train=False,
        download=True,
        transform=None
    )
    real_images = []
    for img, label in mnist_dataset:
        if label == 1:
            real_images.append(img)  # raw PIL
            if len(real_images) == 10:
                break

    #---------------------
    # Step B: Load 10 generated images run1.png ... run10.png
    #---------------------
    gen_filenames = [f"run{i}.png" for i in range(1, 11)]
    gen_images = [Image.open(fname) for fname in gen_filenames]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===============================================================================
    # 1) Inception Score (IS) -- computed ONLY on generated images
    #===============================================================================
    inception_model_is = load_inception_for_is(device)
    is_mean, is_std = compute_inception_score(gen_images, inception_model_is, device, splits=1)
    print(f"Inception Score (10 generated images): {is_mean:.4f} ± {is_std:.4f}")

    #===============================================================================
    # 2) KID (Kernel Inception Distance)
    #===============================================================================
    # We need a second Inception model with fc = Identity() to get 2048-dim embeddings
    inception_model_feats = load_inception_for_features(device)
    
    # Get embeddings for real images (MNIST '1')
    real_acts = get_inception_activations(real_images, inception_model_feats, device)
    # Get embeddings for generated images
    gen_acts  = get_inception_activations(gen_images, inception_model_feats, device)

    # We'll do KID with a polynomial kernel of degree=3
    kid_mean, kid_std = compute_kid(
        real_acts, gen_acts, 
        kernel_fn=lambda a,b: polynomial_kernel(a,b, degree=3, c=1.0),
        splits=5  # do multiple splits for demonstration
    )
    print(f"KID (MNIST '1' vs. generated): {kid_mean:.4f} ± {kid_std:.4f}")

    #===============================================================================
    # 3) FID (Fréchet Inception Distance)
    #===============================================================================
    fid_val = compute_fid(real_acts, gen_acts)
    print(f"FID (MNIST '1' vs. generated): {fid_val:.4f}")


if __name__ == "__main__":
    main()

