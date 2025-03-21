import torch

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