import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
import math
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os
import scipy

# -------------------------------------------------------------------
#  DEVICE AND DATA
# -------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Download the MNIST dataset if not found locally.
data_train = MNIST('.', download=True, train=True)
data_test  = MNIST('.', download=True, train=False)

# Convert from [0..255] to [-1..1].
# Shape after this is (N, 28, 28)
x_train = 2.0 * (data_train.data / 255.0 - 0.5)
x_val   = 2.0 * (data_test.data  / 255.0 - 0.5)
y_train = data_train.targets
y_val   = data_test.targets

# Optionally, pad from 28×28 to 32×32 (value = -1) to match the reference model dimensions.
# This is the only extra step needed to handle 28×28 inside the reference classifier.
x_train = F.pad(x_train, (2,2,2,2), value=-1)  # shape: (N, 32, 32)
x_val   = F.pad(x_val,   (2,2,2,2), value=-1)  # shape: (N, 32, 32)

# For training, the classifier expects shape (N,1,32,32), so add a channels dimension.
x_train = x_train.unsqueeze(1)  # shape: (N, 1, 32, 32)
x_val   = x_val.unsqueeze(1)    # shape: (N, 1, 32, 32)

# -------------------------------------------------------------------
#  CLASSIFIER DEFINITION
# -------------------------------------------------------------------
class Classifier(torch.nn.Module):
    """
    Simple CNN classifier for 32×32 grayscale inputs (MNIST is padded to 32×32).
    The final classification layer is 'classification_layer',
    while 'layers' ends with a dropout output you can treat as a penultimate feature.
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = torch.nn.Sequential(
            # 1×32×32
            torch.nn.Conv2d(1, 8, 3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 8×16×16

            torch.nn.Conv2d(8, 16, 3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2), # 16×8×8

            torch.nn.Conv2d(16, 32, 3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(4, 4), # 32×2×2 => 32×1×1 after avgpool => shape is (N,32)

            torch.nn.Flatten(),       # => shape (N,32)
            torch.nn.Dropout(),       # => shape (N,32)
            # The last layer in `layers` is just the penultimate feature representation.
        )
        # We map that 32-dim feature to 10 classes (digits 0..9).
        self.classification_layer = torch.nn.Linear(128, 10)
        

    def forward(self, x):
        # x assumed shape: (N, 1, 32, 32)
        feat = self.layers(x)                        # (N,32)
        out  = self.classification_layer(feat)       # (N,10)
        return out


# -------------------------------------------------------------------
#  TRAINING THE CLASSIFIER
# -------------------------------------------------------------------
batch_size    = 128
learning_rate = 1e-3
num_epochs    = 20  # Typically you'd do more (e.g., 20+), but adjust as needed.

classifier = Classifier().to(device)
optimizer  = torch.optim.Adam(classifier.parameters(), learning_rate)

# Create DataLoaders for convenience
train_ds = torch.utils.data.TensorDataset(x_train, y_train)
val_ds   = torch.utils.data.TensorDataset(x_val,   y_val)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)  # shape: (B,1,32,32)
        y_batch = y_batch.to(device)

        # Forward
        logits = classifier(x_batch)
        loss   = cross_entropy(logits, y_batch)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    classifier.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch_val, y_batch_val in val_loader:
            x_batch_val = x_batch_val.to(device)
            y_batch_val = y_batch_val.to(device)
            logits_val  = classifier(x_batch_val)
            loss_val    = cross_entropy(logits_val, y_batch_val)
            val_loss   += loss_val.item()

    running_loss /= len(train_loader)
    val_loss     /= len(val_loader)
    print(f"[Epoch {epoch+1}/{num_epochs}] loss={running_loss:.4f}, val_loss={val_loss:.4f}")

# -------------------------------------------------------------------
#  SAVE THE CLASSIFIER CHECKPOINT
# -------------------------------------------------------------------
torch.save(classifier.state_dict(), f"classifier_{num_epochs}epochs.pt")
print("Classifier checkpoint saved to classifier.pt")


# # -------------------------------------------------------------------
# #  EVALUATE & EXTRACT FEATURES FOR FID
# # -------------------------------------------------------------------
# # We'll treat the penultimate layer's output (the "features" before classification_layer)
# # as the embedding for FID. So let's rewrite a small forward that returns that penultimate feature:
# def get_penultimate_features(model, x):
#     # x shape: (N,1,32,32)
#     with torch.no_grad():
#         feats = model.layers(x)  # shape => (N,32)
#     return feats

# classifier.eval()

# # Get embeddings for training data
# train_feats = []
# for x_b, _ in train_loader:
#     x_b = x_b.to(device)
#     feats_b = get_penultimate_features(classifier, x_b)
#     train_feats.append(feats_b.cpu().numpy())
# train_feats = np.concatenate(train_feats, axis=0)

# # Get embeddings for validation data
# val_feats = []
# for x_b, _ in val_loader:
#     x_b = x_b.to(device)
#     feats_b = get_penultimate_features(classifier, x_b)
#     val_feats.append(feats_b.cpu().numpy())
# val_feats = np.concatenate(val_feats, axis=0)

# # Suppose you have a generated dataset "gen_feats" (shape Nx32) from your model
# # or from random data. For illustration, here's how you could create some random features:
# # (In practice, you would feed your real or generated images through 'classifier.layers(...)')
# gen_feats = np.random.randn(val_feats.shape[0], train_feats.shape[1]) * 0.5

# # -------------------------------------------------------------------
# #  FRECHET DISTANCE UTILITY
# # -------------------------------------------------------------------
# def frechet_distance(x_a, x_b):
#     """
#     Compute Fréchet distance between two sets of vectors x_a, x_b
#     x_a shape => (N, d)
#     x_b shape => (M, d)
#     """
#     mu_a    = np.mean(x_a, axis=0)
#     sigma_a = np.cov(x_a.T)
#     mu_b    = np.mean(x_b, axis=0)
#     sigma_b = np.cov(x_b.T)

#     diff = mu_a - mu_b
#     # sqrtm can yield complex values if the covariance matrices are nearly singular.
#     covmean, _ = scipy.linalg.sqrtm(sigma_a @ sigma_b, disp=False)
#     # If there is a tiny imaginary component, just drop it.
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real

#     fd = np.sum(diff**2) + np.trace(sigma_a + sigma_b - 2.0*covmean)
#     return fd

# fd_rand_vs_val = frechet_distance(gen_feats, val_feats)
# fd_train_vs_val = frechet_distance(train_feats, val_feats)
# print(f"FID (random vs val)  = {fd_rand_vs_val:.3f}")
# print(f"FID (train vs val)   = {fd_train_vs_val:.3f}")
