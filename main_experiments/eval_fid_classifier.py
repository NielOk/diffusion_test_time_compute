import torch
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# -------------------------------------------------------------------
#  DEVICE
# -------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------------------------------------------
#  REDEFINE THE SAME CLASSIFIER
# -------------------------------------------------------------------
class Classifier(torch.nn.Module):
    """
    Same CNN classifier as in your training script.
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 8×16×16

            torch.nn.Conv2d(8, 16, 3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2), # 16×8×8

            torch.nn.Conv2d(16, 32, 3, padding='same'),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(4, 4), # 32×2×2 => (N,32)

            torch.nn.Flatten(),
            torch.nn.Dropout(),
        )
        self.classification_layer = torch.nn.Linear(128, 10)

    def forward(self, x):
        feat = self.layers(x)           # shape: (N,32)
        out  = self.classification_layer(feat)  # shape: (N,10)
        return out

def main():
    # -------------------------------------------------------------------
    #  LOAD THE TEST/VAL DATA
    # -------------------------------------------------------------------
    # Download the MNIST test set (set "train=False").
    data_test = MNIST('.', download=True, train=False)
    x_val = 2.0 * (data_test.data / 255.0 - 0.5)  # shape (N,28,28)
    y_val = data_test.targets

    # Pad from 28×28 to 32×32, same as training script
    x_val = F.pad(x_val, (2,2,2,2), value=-1)  # shape (N,32,32)

    # Add channels dimension (N,1,32,32)
    x_val = x_val.unsqueeze(1)

    # Create DataLoader
    val_ds   = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # -------------------------------------------------------------------
    #  LOAD THE CHECKPOINT
    # -------------------------------------------------------------------
    model = Classifier().to(device)
    model.load_state_dict(torch.load('classifier.pt', map_location=device))
    model.eval()

    # -------------------------------------------------------------------
    #  EVALUATE
    # -------------------------------------------------------------------
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch_val, y_batch_val in val_loader:
            x_batch_val = x_batch_val.to(device)
            y_batch_val = y_batch_val.to(device)

            logits = model(x_batch_val)            # (B,10)
            preds  = torch.argmax(logits, dim=1)   # (B,)

            correct += (preds == y_batch_val).sum().item()
            total   += y_batch_val.size(0)

    accuracy = correct / total
    print(f"Accuracy on test set: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
