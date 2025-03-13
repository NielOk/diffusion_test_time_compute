import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import scipy.linalg

CLASSIFIER_CHECKPOINT = 'classifier_160epochs.pt'
IMAGE_FOLDER = "./noise_sample_images"


# 1) DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2) RE‐DEFINE CLASSIFIER (SAME ARCHITECTURE)
class Classifier(torch.nn.Module):
    """
    Simple CNN classifier for 32×32 grayscale inputs.
    The final classification layer is 'classification_layer',
    while 'layers' ends with a dropout output you can treat as a penultimate feature.
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = torch.nn.Sequential(
            # 1×32×32
            torch.nn.Conv2d(1, 8, 3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # => 8×16×16

            torch.nn.Conv2d(8, 16, 3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2), # => 16×8×8

            torch.nn.Conv2d(16, 32, 3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(4, 4), # => shape (N, 32×1×1) = (N,32)

            torch.nn.Flatten(),       # => shape (N,32)
            torch.nn.Dropout(),       # => shape (N,32)
            # The last layer in `layers` is our penultimate feature representation
        )
        # We map that 32-dim feature to 10 classes (digits 0..9).
        self.classification_layer = torch.nn.Linear(128, 10)

    def forward(self, x):
        # x shape: (N, 1, 32, 32)
        feat = self.layers(x)                  # => (N, 32)
        out  = self.classification_layer(feat) # => (N, 10)
        return out

def get_penultimate_features(model, x):
    """
    Returns the 32‐D penultimate features before final classification.
    """
    with torch.no_grad():
        feats = model.layers(x)  # shape => (N, 32)
    return feats

# 3) LOAD CLASSIFIER CHECKPOINT
classifier = Classifier().to(device)
classifier.load_state_dict(torch.load(CLASSIFIER_CHECKPOINT, map_location=device))
classifier.eval()
print(f"Loaded classifier from {CLASSIFIER_CHECKPOINT}")

# 4) PREPARE REFERENCE EMBEDDINGS FROM THE VALIDATION SET
#    (If you'd rather compare to training set, just do similarly with x_train, y_train.)
#    -- The easiest approach is to re-run exactly the snippet from your training script
#       that loads MNIST test set, pads, and accumulates val_feats.

# But let's do a minimal re-implementation for clarity:
from torchvision.datasets import MNIST

data_test  = MNIST('.', download=True, train=False)   # shape (N, 28, 28)
x_val = 2.0 * (data_test.data / 255.0 - 0.5)          # => [-1..1], shape (N,28,28)
x_val = F.pad(x_val, (2,2,2,2), value=-1)             # => (N,32,32)
x_val = x_val.unsqueeze(1)                            # => (N,1,32,32)

val_targets = data_test.targets                       # not strictly needed for FID
val_ds   = torch.utils.data.TensorDataset(x_val, val_targets)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

val_feats = []
for x_b, _ in val_loader:
    x_b = x_b.to(device)
    feats_b = get_penultimate_features(classifier, x_b)
    val_feats.append(feats_b.cpu().numpy())
val_feats = np.concatenate(val_feats, axis=0)  # shape => (N_val, 32)
print("Computed reference val_feats shape:", val_feats.shape)

# 5) CREATE A DATASET FOR YOUR GENERATED IMAGES FOLDER
#    We'll assume the folder has images all in grayscale, 28x28, naming doesn't matter.
#    This dataset will:
#       - load images in grayscale
#       - convert to tensor
#       - scale [0..1] => [-1..1]
#       - pad to (32,32) with -1
#    Then you can feed them in batches to the classifier to get features.

class GeneratedMNISTDataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        self.image_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),   # ensure 1‐channel
            T.ToTensor(),                         # => FloatTensor in [0..1], shape (1,H,W)
            # We'll do the [-1..1] scaling manually below
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path)
        img = self.transform(img)  # => shape (1,H,W), values in [0..1]
        
        # Scale [0,1] => [-1,1]
        img = 2.0 * (img - 0.5)
        
        # Now pad from 28×28 to 32×32 with value=-1 (assuming exactly 28×28)
        # If your generated images are already 32×32, remove this padding step.
        pad = (2,2,2,2)  # left,right,top,bottom
        img = F.pad(img, pad, value=-1)  # => shape (1,32,32)

        return img

# 6) GET PENULTIMATE FEATURES FOR THE GENERATED IMAGES
def get_features_for_folder(model, folder, batch_size=256, num_workers=4):
    """
    Load all images in `folder`, process in batches on GPU,
    return an (N,32) array of penultimate features.
    """
    dataset = GeneratedMNISTDataset(folder)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    
    all_feats = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)            # shape (B,1,32,32)
            feats = get_penultimate_features(model, batch)  # => (B,32)
            all_feats.append(feats.cpu().numpy())
    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats

# 7) FRECHET DISTANCE (FID) FUNCTION
def frechet_distance(x_a, x_b):
    """
    Compute Fréchet distance between two sets of vectors x_a, x_b
    x_a shape => (N, d)
    x_b shape => (M, d)
    """
    mu_a    = np.mean(x_a, axis=0)
    sigma_a = np.cov(x_a.T)
    mu_b    = np.mean(x_b, axis=0)
    sigma_b = np.cov(x_b.T)

    diff = mu_a - mu_b
    # sqrtm can yield complex values if the covariance matrices are nearly singular.
    covmean, _ = scipy.linalg.sqrtm(sigma_a @ sigma_b, disp=False)
    # If there is a tiny imaginary component, just drop it.
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = np.sum(diff**2) + np.trace(sigma_a + sigma_b - 2.0*covmean)
    return fd

# 8) MAIN: compute FID from generated folder vs. val_feats
if __name__ == "__main__":
    np.random.seed(42) # I'm using this for reproducibility to compare FID across different runs of this script
        
    generated_folder = IMAGE_FOLDER  # <-- change to your folder path
    gen_feats = get_features_for_folder(classifier, generated_folder,
                                        batch_size=256, num_workers=4)
    print("gen_feats shape:", gen_feats.shape)
    
    n_val = val_feats.shape[0]
    n_gen = gen_feats.shape[0]
    
    n = min(n_val, n_gen)
    if n_gen != n_val:
        print(f"Using {n} images from each set (generated: {n_gen}, validation: {n_val}).")

    indices_val = np.random.choice(n_val, n, replace=False)
    indices_gen = np.random.choice(n_gen, n, replace=False)
    val_feats_sampled = val_feats[indices_val]
    gen_feats_sampled = gen_feats[indices_gen]

    # Compute FID
    fid = frechet_distance(val_feats_sampled, gen_feats_sampled)
    print(f"FID (generated vs val) = {fid:.4f}")
