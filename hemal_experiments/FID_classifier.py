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
x_train = data_train.data.float() / 255.0 * 2.0 - 1.0
x_val   = data_test.data.float()  / 255.0 * 2.0 - 1.0
y_train = data_train.targets
y_val   = data_test.targets

x_train = x_train.unsqueeze(1)  # shape: (N, 1, 28, 28)
x_val   = x_val.unsqueeze(1)    # shape: (N, 1, 28, 28)

# -------------------------------------------------------------------
#  CLASSIFIER DEFINITION
# -------------------------------------------------------------------
class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.features = torch.nn.Sequential(
            # First block
            torch.nn.Conv2d(1, 32, 3, padding=1),    # (N, 32, 28, 28)
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),                # (N, 32, 14, 14)

            # Second block
            torch.nn.Conv2d(32, 64, 3, padding=1),   # (N, 64, 14, 14)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),                # (N, 64, 7, 7)

            # Third block
            torch.nn.Conv2d(64, 128, 3, padding=1),  # (N, 128, 7, 7)
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),      # (N, 128, 1, 1)
            torch.nn.Flatten(),                      # (N, 128)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# -------------------------------------------------------------------
#  TRAINING THE CLASSIFIER
# -------------------------------------------------------------------
batch_size    = 32 # Make sure this matches the batch size used in the classifier
learning_rate = 1e-3
num_epochs    = 20 # Typically you'd do more (e.g., 20+), but adjust as needed.

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
        logits = classifier(x_batch)
        loss   = cross_entropy(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    classifier.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch_val, y_batch_val in val_loader:
            logits_val = classifier(x_batch_val)
            loss_val = cross_entropy(logits_val, y_batch_val)
            val_loss += loss_val.item()

            preds = torch.argmax(logits_val, dim=1)
            correct += (preds == y_batch_val).sum().item()
            total   += y_batch_val.size(0)

    running_loss /= len(train_loader)
    val_loss     /= len(val_loader)
    val_acc      = 100.0 * correct / total

    print(f"[Epoch {epoch+1:02}/{num_epochs}] "
          f"loss={running_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")

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
