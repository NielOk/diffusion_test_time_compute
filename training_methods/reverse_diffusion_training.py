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
from tqdm import tqdm
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
    
# Improved convolutional network for reverse diffusion
class ImprovedDiffuser(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, base_channels=64):
        super(ImprovedDiffuser, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(base_channels)  # Normalize activations
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(base_channels)
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)  

    def forward(self, x):
        res = self.residual(x)  # Shortcut
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        return x + res  # Add residual connection
    
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

# Simple UNet
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, base_channels)
        self.enc2 = UNetBlock(base_channels, base_channels * 2)
        self.dec2 = UNetBlock(base_channels * 2, base_channels)
        self.dec1 = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        d2 = self.dec2(F.interpolate(e2, scale_factor=2))
        d1 = self.dec1(d2 + e1)  # Skip connection
        return d1
    
def load_non_noisy_data(training_data_path):
    generator = DynamicForwardDiffuser()

    generator.load_training_data(training_data_path)
    merged_data_list, merged_data_labels = generator.convert_data_to_neural_net_format()

    # Convert to arrays
    merged_data_array = np.stack(merged_data_list)
    merged_labels_array = np.array(merged_data_labels)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(merged_data_array, merged_labels_array, test_size=0.2, random_state=42)

    batch_size = 4
    X_train_batches = generator.create_batches(np.array(X_train), batch_size)
    X_test_batches = generator.create_batches(np.array(X_test), batch_size)
    y_train_batches = generator.create_batches(np.array(y_train), batch_size)
    y_test_batches = generator.create_batches(np.array(y_test), batch_size)

    return X_train_batches, X_test_batches, y_train_batches, y_test_batches, generator

def train_model(model, X_train_batches, y_train_batches, generator, beta, num_diffusion_steps=20, num_epochs=10, learning_rate=0.0001, model_save_filename='simple_conv_net_diffuser.pth'):
    ForwardDiffuser = DynamicForwardDiffuser() # Set up the dynamic forward diffuser

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        
        model.train() # Set model to training mode

        for i in range(len(X_train_batches)):

            # Forward diffusion on inputs to get noisy data
            batch_steps, prev_step_noises = generator.batch_beta_schedule_forward_diffusion(X_train_batches[i], num_diffusion_steps, beta)

            # Convert labels to torch tensor float
            labels = torch.from_numpy(y_train_batches[i]).float()

            # Get the embeddings and normalize
            batch_steps = ForwardDiffuser.embed_and_normalize(batch_steps, y_train_batches[i], num_diffusion_steps)

            # Convert batch steps to torch tensor floats, normalize, add label embedding, time step embeddings
            for key, value in batch_steps.items():

                if key != 0: # Skip the non-noisy data

                    batch_steps[key] = torch.from_numpy(value).float() # Convert to torch tensor float

                    prev_step_noises[key] = torch.from_numpy(prev_step_noises[key]).float()
                    prev_step_noises[key] = prev_step_noises[key].permute(0, 3, 1, 2) # Change to fit the model input shape
                    
                    model_input = batch_steps[key].permute(0, 3, 1, 2) # Change to fit the model input shape
                    
                    output = model(model_input)
                    loss = criterion(output, prev_step_noises[key]) # Loss calculation

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        torch.save(model.state_dict(), os.path.join(TRAINING_METHODS_DIR, model_save_filename)) # Save model after each epoch

def test_model(model, X_test_batches, y_test_batches, generator, beta, num_diffusion_steps=20):
    ForwardDiffuser = DynamicForwardDiffuser() # Set up the dynamic forward diff
    criterion = torch.nn.MSELoss()

    model.eval() # Set model to evaluation mode

    for i in range(len(X_test_batches)):

        # Forward diffusion on inputs to get noisy data. The prev_step_noise of step 1 is prev_step_noise[1], is noise added at step 0
        batch_steps, prev_step_noises = generator.batch_beta_schedule_forward_diffusion(X_test_batches[i], num_diffusion_steps, beta)

        # Convert labels to torch tensor float
        labels = torch.from_numpy(y_test_batches[i]).float()

        # Get the embeddings and normalize
        batch_steps = ForwardDiffuser.embed_and_normalize(batch_steps, y_test_batches[i], num_diffusion_steps)

        # Convert batch steps to torch tensor floats, normalize, add label embedding, time step embeddings
        for key, value in batch_steps.items():

            if key != 0: # Skip the non-noisy data

                batch_steps[key] = torch.from_numpy(value).float() # Convert to torch tensor float

                prev_step_noises[key] = torch.from_numpy(prev_step_noises[key]).float()
                prev_step_noises[key] = prev_step_noises[key].permute(0, 3, 1, 2) # Change to fit the model input shape
                
                model_input = batch_steps[key].permute(0, 3, 1, 2) # Change to fit the model input shape

                output = model(model_input)
                loss = criterion(output, prev_step_noises[key])

                print(f"Step: {key}, Loss: {loss}")

def visualize_model_output(model, beta, num_diffusion_steps=20):
    ForwardDiffuser = DynamicForwardDiffuser() # Set up the dynamic forward diffuser, just for its embedding functions
    
    model.eval() # Set model to evaluation mode

    # Get alpha and alpha_bar
    alpha = 1.0 - beta

    original_image_input = np.random.normal(0, 1, (1, 32, 32, 3))  # Gaussian noise
    original_image_input = np.clip(original_image_input, -1, 1)
    original_image_input = ((original_image_input + 1) / 2)

    # Show the original image

    # normalize to be between 0 and 255
    image_array = original_image_input * 255
    image_array = image_array.squeeze().astype(np.uint8)
    image = Image.fromarray(image_array)
    image.show()

    image_input = original_image_input
    for diffusion_step in reversed(range(num_diffusion_steps)):

        if diffusion_step == 0:
            break # We add no noise before the zero step

        # Get the embeddings and normalize
        le = ForwardDiffuser.label_embedding(np.array([0]), image_input.shape)
        pe = ForwardDiffuser.sinusoidal_positional_embedding(diffusion_step, image_input.shape)

        pe_concatenated_image = np.concatenate([image_input, pe], axis=-1)
        le_pe_concatenated_image = np.concatenate([pe_concatenated_image, pe], axis=-1)

        model_input = torch.from_numpy(le_pe_concatenated_image).float().permute(0, 3, 1, 2) # Change to fit the model input shape

        predicted_noise = model(model_input)

        # Get predicted noise as numpy array
        predicted_noise_array = predicted_noise.detach().numpy().squeeze().transpose(1, 2, 0)

        # Get the next image
        image_input = (1 / np.sqrt(alpha[diffusion_step])) * (image_input - np.sqrt(1 - alpha[diffusion_step]) * predicted_noise_array)

        #Draw the noisy image
        image_array_2 = np.clip(image_input, -1, 1)

        # normalize to be between 0 and 1
        image_array_2 = ((image_array_2 + 1) / 2) * 255
        image_array_2 = image_array_2.squeeze().astype(np.uint8)
        image2 = Image.fromarray(image_array_2)
        image2.show()

# Example usage
if __name__ == "__main__":

    # Load training data
    training_data_path = os.path.join(TRAINING_DATA_METHODS_DIR, 'training_data.json')
    X_train_batches, X_test_batches, y_train_batches, y_test_batches, generator = load_non_noisy_data(training_data_path)

    # Define beta
    T = 150
    beta = np.linspace(0.0001, 0.02, T)  # Uniform beta schedule

    # Train model
    model_save_filename = 'simple_conv_net_diffuser.pth'
    model = SimpleConvNetDiffuser()
    train_model(model, X_train_batches, y_train_batches, generator, beta, num_diffusion_steps=T, num_epochs=20, learning_rate=0.001, model_save_filename=model_save_filename)

    # Load trained model 
    trained_model = SimpleConvNetDiffuser()
    model_save_dir = os.path.join(TRAINING_METHODS_DIR, model_save_filename)
    trained_model.load_state_dict(torch.load(model_save_dir))

    # Test model
    test_model(trained_model, X_test_batches, y_test_batches, generator, beta, num_diffusion_steps=T)

    # Visualize model output
    visualize_model_output(trained_model, beta, num_diffusion_steps=T)