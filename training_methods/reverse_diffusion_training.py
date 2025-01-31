'''
Code for reverse diffuser and training.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import sys
from PIL import Image
from sklearn.model_selection import train_test_split

TRAINING_METHODS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TRAINING_METHODS_DIR)
TRAINING_DATA_METHODS_DIR = os.path.join(REPO_DIR, 'training_data_methods')

sys.path.append(TRAINING_DATA_METHODS_DIR)
from dynamic_forward_diffuser import DynamicForwardDiffuser

# Simple convolutional network for reverse diffusion
class SimpleConvNetDiffuser(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, base_channels=64):
        super(SimpleConvNetDiffuser, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
def load_non_noisy_data(training_data_path):
    generator = DynamicForwardDiffuser()

    generator.load_training_data(training_data_path)
    merged_data_list, merged_data_labels = generator.convert_data_to_neural_net_format()

    # Convert to arrays
    merged_data_array = np.stack(merged_data_list)
    merged_labels_array = np.array(merged_data_labels)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(merged_data_array, merged_labels_array, test_size=0.2, random_state=42)

    batch_size = 8
    X_train_batches = generator.create_batches(np.array(X_train), batch_size)
    X_test_batches = generator.create_batches(np.array(X_test), batch_size)
    y_train_batches = generator.create_batches(np.array(y_train), batch_size)
    y_test_batches = generator.create_batches(np.array(y_test), batch_size)

    return X_train_batches, X_test_batches, y_train_batches, y_test_batches, generator

def train_model(model, X_train_batches, y_train_batches, generator, beta, num_diffusion_steps=20, num_epochs=100, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        for i in range(len(X_train_batches)):

            # Forward diffusion on inputs to get noisy data
            batch_steps = generator.batch_beta_schedule_forward_diffusion(X_train_batches[i], num_diffusion_steps, beta)

            # Convert labels to torch tensor float
            labels = torch.from_numpy(y_train_batches[i]).float()

            # Convert batch steps to torch tensor floats
            for key, value in batch_steps.items():
                batch_steps[key] = torch.from_numpy(value).float()

    
# Example usage
if __name__ == "__main__":

    # Load training data
    training_data_path = os.path.join(TRAINING_DATA_METHODS_DIR, 'training_data.json')
    X_train_batches, X_test_batches, y_train_batches, y_test_batches, generator = load_non_noisy_data(training_data_path)

    # Define beta
    T = 20
    beta = np.linspace(0.0001, 0.02, T)  # Uniform beta schedule

    # Train model
    model = SimpleConvNetDiffuser()
    train_model(model, X_train_batches, y_train_batches, generator, beta, num_diffusion_steps=T, num_epochs=1, learning_rate=0.001)