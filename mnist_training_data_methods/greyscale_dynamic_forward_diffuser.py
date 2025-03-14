'''
This file contains a class that generates forward diffusion training data for greyscale images dynamically.
'''

from PIL import Image
import numpy as np
from typing import Tuple, Dict, List
import json
from torchvision import datasets

class GreyscaleDynamicForwardDiffuser:
     
    def __init__(self) -> None:
        self.non_noisy_data = None # Will be used to store the non-noisy image data
        self.non_noisy_labels = None # Will be used to store the non-noisy image labels

    def load_training_data(self, 
                        training_data_path: str, # Path to the training data
                        shuffle: bool = True, # Whether to shuffle the data
                        download: bool = True # Whether to download the data
                        ) -> None:
          
        mnist_pytorch = datasets.MNIST(root=training_data_path, train=True, download=download)

        mnist_data = mnist_pytorch.data.numpy()

        mnist_data = np.expand_dims(mnist_data, axis=-1) # Add a channel dimension for time and label embeddings
        mnist_labels = mnist_pytorch.targets.numpy()

        self.non_noisy_data = mnist_data
        self.non_noisy_labels = mnist_labels

    def create_batches(self, 
                    x_data: np.ndarray, # The data to create batches from
                    y_data: np.ndarray, # The labels to create batches from
                    batch_size: int, # Size of each batch
                    shuffle: bool = True # Whether to shuffle the data
                    ) -> List[np.ndarray]:
                
        '''
        Create batches from a data array. This is to be used for both
        features and labels (the batch size should be the same for both 
        for separate datasets).
        '''
        num_samples = x_data.shape[0]

        if shuffle:
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_data, y_data = x_data[indices], y_data[indices]
        
        # Split data into batches
        data_batches = np.array_split(x_data, np.ceil(num_samples / batch_size))
        label_batches = np.array_split(y_data, np.ceil(num_samples / batch_size))
        
        return data_batches, label_batches
    
    def sinusoidal_positional_embedding(self,
                             t: int, # Current time step
                             shape: Tuple[int, int, int, int] # Shape of the image (batch size, height, width, channels). Number of channels for the embedding is always 1, no matter what the image is
                             ) -> np.ndarray:
        '''
        Generate a sinusoidal positional embedding for the current time 
        step t of shape (batch size, height, width, channels). 
        ''' 

        batch_size, height, width, _ = shape

        # compute a single embedding scalar for the time step `t`
        embedding_value = np.sin(t / 10000)

        # create a tensor of shape (batch size, height, width, 1) with the same value
        embedding_tensor = np.full((batch_size, height, width, 1), embedding_value, dtype=np.float32)

        return embedding_tensor
    
    def label_embedding(self,
            label_ids: np.ndarray, # The labels to embed
            shape: Tuple[int, int, int, int] # The shape of the image tensor
            ) -> np.ndarray:
        '''
        Generate a label embedding for the given label_id of shape
        '''
        batch_size, height, width, _ = shape

        # Compute embedding values to be the label ids with float32 type
        embedding_values = label_ids.astype(np.float32)

        # Create a tensor of shape (batch_size, height, width, 1)
        embedding_tensor = np.tile(embedding_values[:, np.newaxis, np.newaxis, np.newaxis], 
                                (1, height, width, 1))

        return embedding_tensor
    
    def batch_uniform_scaled_forward_diffusion(self, 
                                               batch_array: np.ndarray, # Array of all non-noisy data
                                               T: int # Number of diffusion steps
                                               ) -> Dict[int, np.ndarray]: # Returns a dictionary with the step number as the key and the batch array as the value
        '''
        Take a batch of image tensors, apply forward diffusion to 
        each each batch, and return a dictionary with the step 
        number as the keys and the batch arrays as the values.
        '''

        batch_T = (batch_array.astype(np.float32) / 127.5) - 1 # Scale the batch to be between -1 and 1 for forward diffusion

        batch_steps = {}
        batch_steps[0] = batch_array

        prev_step_noises = {}

        for t in range(1, T):
            noise = np.random.normal(0, (t + 1) / T, size=batch_T.shape)
            
            batch_T = batch_T + noise
            batch_T = np.clip(batch_T, -1, 1) # Clip the batch to be between -1 and 1

            # Convert the batch back to a numpy array of integers
            noisy_step = np.clip(((batch_T + 1) * 127.5).astype(np.uint8), 0, 255) # Scale the batch back to be between 0 and 255

            batch_steps[t] = noisy_step
            prev_step_noises[t] = noise

        return batch_steps, prev_step_noises
    
    def batch_beta_schedule_forward_diffusion(self,
                                        batch_array: np.ndarray, # Array of all non-noisy data
                                        T: int, # Number of diffusion steps
                                        beta: np.ndarray # Beta schedule
                                        ) -> Dict[int, np.ndarray]: # Returns a dictionary with the step number as the key and the batch array as the value
        '''
        Do forward diffusion with a uniform beta schedule.
        '''

        batch_T = (batch_array.astype(np.float32) / 127.5) - 1 # Scale the batch to be between -1 and 1 for forward diffusion

        batch_steps = {}
        batch_steps[0] = batch_array

        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha)  # Cumulative product of alphas

        prev_step_noises = {}
        
        for t in range(1, T):
            noise = np.random.normal(0, 1, batch_T.shape)  # Gaussian noise
            batch_T = np.sqrt(alpha_bar[t]) * batch_T + np.sqrt(1 - alpha_bar[t]) * noise
            batch_T = np.clip(batch_T, -1, 1)  # Clip the batch to be between -1 and 1

            # Convert the batch back to a numpy array of integers
            noisy_step = np.clip(((batch_T + 1) * 127.5).astype(np.uint8), 0, 255)  # Scale the batch back to be between 0 and 255

            batch_steps[t] = noisy_step
            prev_step_noises[t] = noise # Save the noise for training
        
        return batch_steps, prev_step_noises
    
    def embed_and_normalize(self, 
                        batch_steps: Dict[int, np.ndarray], # Dictionary with the step number as the key and the batch array as the value
                        batch_labels: np.ndarray, # Array of all labels
                        T: int # Number of diffusion steps
                        ) -> Dict[int, np.ndarray]: # Returns a dictionary with the step number as the key and the batch array as the value
        '''
        Add all embeddings to the batch of diffusion steps.
        '''

        for t in range(T):
            normalized_batch = batch_steps[t].astype(np.float32) / 255.0
            le = self.label_embedding(batch_labels, batch_steps[t].shape)
            pe = self.sinusoidal_positional_embedding(t, batch_steps[t].shape)
            batch_steps[t] = np.concatenate([normalized_batch, pe], axis=-1)
            batch_steps[t] = np.concatenate([batch_steps[t], le], axis=-1)

        return batch_steps