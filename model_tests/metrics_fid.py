###############################################################################
# fid_only_demo_5.py
#
# A self-contained Python script demonstrating how to compute FID on MNIST
# with a toy "Inception-like" model that operates on 1×28×28 images.
# We only compute FID -- no Inception Score or KID.
#
# Usage:
#   python fid_only_demo_5.py
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
        x = x.view(b, -1)            # shape [b, 784]
        feats = self.features_layer(x)    # shape [b, 128]
        logits = self.classifier(feats)   # shape [b, 10]
        return logits, feats

    def features(self, x):
        """
        Returns the penultimate-layer (128-D) embeddings.
        """
        _, feats = self.forward(x)
        return feats


###############################################################################
# 2) Utility: extract penultimate-layer features from PIL images
###############################################################################

def get_mnist_features(images, model, device):
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
            feats = model.features(img_t)  # shape [1,128]
        feats_list.append(feats.squeeze(0).cpu().numpy())
    return np.vstack(feats_list)


###############################################################################
# 3) FID
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
# 4) Main: compute FID on 5 real vs. 5 "generated" MNIST '5' images
###############################################################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create + eval mode
    mnist_inception_model = MnistInceptionModel().to(device)
    mnist_inception_model.eval()

    #---------------------------------------------------------------------------
    # A) Gather 5 real MNIST '5' images
    #---------------------------------------------------------------------------
    mnist_test = torchvision.datasets.MNIST(
        root='./mnist_data',
        train=False,
        download=True,
        transform=None
    )
    real_images = []
    for img, label in mnist_test:
        if label == 5:
            real_images.append(img)  # raw PIL
            if len(real_images) == 5:
                break

    #---------------------------------------------------------------------------
    # B) Gather 5 "generated" images (just picking 5 more from class '5')
    #---------------------------------------------------------------------------
    gen_images = []
    count = 0
    for img, label in mnist_test:
        if label == 5:
            count += 1
            if count >= 30 and count < 35:
                gen_images.append(img)
            if len(gen_images) == 5:
                break

    #---------------------------------------------------------------------------
    # 1) Extract penultimate-layer features
    #---------------------------------------------------------------------------
    real_feats = get_mnist_features(real_images, mnist_inception_model, device)
    gen_feats  = get_mnist_features(gen_images,  mnist_inception_model, device)

    #---------------------------------------------------------------------------
    # 2) Compute FID
    #---------------------------------------------------------------------------
    fid_val = compute_fid(real_feats, gen_feats)
    print(f"\n==> FID (5 real '5' vs 5 generated '5') = {fid_val:.4f}")


if __name__ == "__main__":
    main()
