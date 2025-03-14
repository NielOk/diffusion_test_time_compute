###############################################################################
# metrics_demo.py
#
# A self-contained Python script demonstrating:
#   - Inception Score (MNIST version)
#   - FID
#   - KID
# with a toy "MNIST Inception" model that operates on 1×28×28 images.
#
# Usage:
#   python metrics_demo.py
#
###############################################################################

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm

###############################################################################
# 1) Define a toy "MNIST Inception" model
###############################################################################

class MnistInceptionModel(torch.nn.Module):
    """
    Minimal "Inception-like" model for MNIST, purely for demonstration.
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
        x = x.view(b, -1)            # shape [b, 784]
        feats = self.features_layer(x)    # shape [b, 128]
        logits = self.classifier(feats)   # shape [b, 10]
        return logits, feats

    def predict_proba(self, x):
        """
        Returns the softmax class probabilities (10-D).
        """
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)

    def features(self, x):
        """
        Returns the penultimate-layer (128-D) embeddings.
        """
        _, feats = self.forward(x)
        return feats


###############################################################################
# 2) Inception Score (MNIST version)
###############################################################################

def compute_inception_score_mnist(images, model, device, splits=1):
    """
    Compute "Inception Score" using the custom MNIST model. 
    images : list of PIL images
    model  : model returning p(y|x) for 10 classes
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # force single-channel
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    probs_list = []
    for img in images:
        # Preprocess to shape [1,1,28,28], send to device
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            p_yx = model.predict_proba(img_t)  # shape [1,10]
        probs_list.append(p_yx.cpu().numpy())

    probs_array = np.vstack(probs_list)  # shape [N,10]
    N = probs_array.shape[0]
    split_size = N // splits

    is_values = []
    for i in range(splits):
        part = probs_array[i*split_size : (i+1)*split_size]
        p_y = np.mean(part, axis=0, keepdims=True)  # shape [1,10]
        
        # KL divergence: E_x[ KL(p(y|x) || p(y)) ]
        kl_divs = part * (np.log(part + 1e-16) - np.log(p_y + 1e-16))
        kl_mean = np.mean(np.sum(kl_divs, axis=1))
        is_values.append(np.exp(kl_mean))

    return float(np.mean(is_values)), float(np.std(is_values))


###############################################################################
# 3) Feature Extraction for FID / KID
###############################################################################

def get_mnist_features(images, model, device):
    """
    Extract 128-D penultimate-layer features from the custom MNIST model.
    images: list of PIL Images
    returns: np.array shape [N,128]
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
            feats = model.features(img_t)  # shape [1,128]
        feats_list.append(feats.squeeze(0).cpu().numpy())
    return np.vstack(feats_list)


###############################################################################
# 4) FID
###############################################################################

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

    fid_val = diff_sq + np.trace(sigma_x + sigma_y - 2*cov_sqrt)
    return fid_val


###############################################################################
# 5) KID (via MMD w/ a polynomial kernel)
###############################################################################

def polynomial_kernel(x, y, degree=3, c=1.0):
    """
    K(x,y) = ((x·y)/dim + c)^degree
    """
    dot = x @ y.T
    dim = x.shape[1]
    return (dot / dim + c) ** degree

def compute_mmd(x, y, kernel_fn):
    Kxx = kernel_fn(x, x)
    Kyy = kernel_fn(y, y)
    Kxy = kernel_fn(x, y)
    n = x.shape[0]
    m = y.shape[0]

    sum_Kxx = np.sum(Kxx) - np.sum(np.diag(Kxx))
    sum_Kyy = np.sum(Kyy) - np.sum(np.diag(Kyy))
    mmd_xx = sum_Kxx / (n*(n-1))
    mmd_yy = sum_Kyy / (m*(m-1))
    mmd_xy = 2.0 * np.sum(Kxy) / (n*m)
    return mmd_xx + mmd_yy - mmd_xy

def compute_kid(x, y, kernel_fn, splits=1):
    rng = np.random.default_rng(123)  # fixed seed for reproducibility
    kid_vals = []
    for _ in range(splits):
        x_perm = rng.permutation(x)
        y_perm = rng.permutation(y)
        mmd2 = compute_mmd(x_perm, y_perm, kernel_fn)
        kid_vals.append(mmd2)
    return float(np.mean(kid_vals)), float(np.std(kid_vals))


###############################################################################
# 6) Main
###############################################################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create + eval mode
    mnist_inception_model = MnistInceptionModel().to(device)
    mnist_inception_model.eval()

    #---------------------------------------------------------------------------
    # A) Gather 10 real MNIST '1' images
    #---------------------------------------------------------------------------
    mnist_test = torchvision.datasets.MNIST(
        root='./mnist_data',
        train=False,
        download=True,
        transform=None
    )
    real_images = []
    for img, label in mnist_test:
        if label == 1:
            real_images.append(img)  # raw PIL
            if len(real_images) == 10:
                break

    #---------------------------------------------------------------------------
    # B) Gather 10 "generated" images (for demonstration, we'll just pick 10 more)
    #    In practice, you'd load your own .png/.jpg files with PIL.Image.open().
    #---------------------------------------------------------------------------
    gen_images = []
    count = 0
    for img, label in mnist_test:
        if label == 1:
            count += 1
            if count >= 30 and count < 40:  # just pick a different slice of "1"s
                gen_images.append(img)
            if len(gen_images) == 10:
                break

    #---------------------------------------------------------------------------
    # 1) Inception Score on the generated set
    #---------------------------------------------------------------------------
    is_mean, is_std = compute_inception_score_mnist(gen_images, mnist_inception_model, device, splits=1)
    print(f"\n==> Inception Score (MNIST) on 10 generated images: {is_mean:.4f} ± {is_std:.4f}")

    #---------------------------------------------------------------------------
    # 2) FID and KID between real vs. generated (both '1's in this toy example)
    #---------------------------------------------------------------------------
    real_feats = get_mnist_features(real_images, mnist_inception_model, device)
    gen_feats  = get_mnist_features(gen_images, mnist_inception_model, device)

    fid_val = compute_fid(real_feats, gen_feats)
    print(f"==> FID (10 real '1' vs 10 generated) = {fid_val:.4f}")

    kid_mean, kid_std = compute_kid(
        real_feats,
        gen_feats,
        kernel_fn=lambda a,b: polynomial_kernel(a,b, degree=3, c=1.0),
        splits=5
    )
    print(f"==> KID (10 real '1' vs 10 generated) = {kid_mean:.4f} ± {kid_std:.4f}\n")


if __name__ == "__main__":
    main()



