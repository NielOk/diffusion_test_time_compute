import torch
import os
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict

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
    
if __name__ == '__main__':
    # Test the classifier if this file is run directly
    model = Classifier()

    future_work_demos_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(future_work_demos_dir)
    hemal_experiments_dir = os.path.join(repo_dir, "hemal_experiments")
    mnist_classifier_160_epochs_path = os.path.join(hemal_experiments_dir, "classifier_20epochs.pt")

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model weights
    model.load_state_dict(torch.load(mnist_classifier_160_epochs_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  # Returns (C, H, W) with pixel values in [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])


    test_dataset = MNIST(root='./test_data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Set up accuracy tracking
    correct_per_digit = defaultdict(int)
    total_per_digit = defaultdict(int)

    # Evaluate on test set
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for label, pred in zip(labels, preds):
            total_per_digit[int(label)] += 1
            if label == pred:
                correct_per_digit[int(label)] += 1

    # Print accuracy per digit
    print("Per-digit accuracy:")
    for digit in range(10):
        correct = correct_per_digit[digit]
        total = total_per_digit[digit]
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        print(f"Digit {digit}: {accuracy:.2f}% ({correct}/{total})")