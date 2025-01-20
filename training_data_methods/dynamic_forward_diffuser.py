'''
This file contains a class that generates forward diffusion training data dynamically.
'''

from PIL import Image
import numpy as np
from typing import Tuple, Dict, List
import json

class DynamicForwardDiffuser:
     
    def __init__(self) -> None:
        self.non_noisy_data = None # Will be used to store the non-noisy data

    def load_training_data(self, 
                        training_data_path: str
                        ) -> None:
          
        with open(training_data_path, 'r') as f:
            self.non_noisy_data = json.load(f)   

    def convert_data_to_neural_net_format(self, 
                                        ) -> Tuple[List[np.ndarray], List[int]]: # Features and what their label is to condition on
        
        '''
        Convert the non-noisy data dict obtained from load_training_data 
        into a tensor of image data and a tensor of labels
        '''
        merged_data_list = []
        merged_data_labels = [] # label as in what the data is conditioned on

        for shape, data_list in self.non_noisy_data.items():
            for data in data_list:
                non_noisy_matrix = np.array(data)
                image_array = non_noisy_matrix.astype(np.uint8)

                merged_data_list.append(image_array)
                merged_data_labels.append(0 if shape == 'square' else 1) # 0 for square, 1 for triangle. 

        return merged_data_list, merged_data_labels

    def create_batches(self, 
                    data: np.ndarray, # Array of all non-noisy data
                    batch_size: int # Size of each batch
                    ) -> List[np.ndarray]:
                
        '''
        Create batches from a data array. This is to be used for both
        features and labels (the batch size should be the same for both 
        for separate datasets).
        '''
        num_samples = data.shape[0]
        # Compute the number of batches
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # Use slicing to create batches
        batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        return batches

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

        for t in range(T - 1):
            noise = np.random.normal(0, (t + 1) / T, size=batch_T.shape)
            
            batch_T = batch_T + noise
            batch_T = np.clip(batch_T, -1, 1) # Clip the batch to be between -1 and 1

            # Convert the batch back to a numpy array of integers
            noisy_step = np.clip(((batch_T + 1) * 127.5).astype(np.uint8), 0, 255) # Scale the batch back to be between 0 and 255

            batch_steps[t + 1] = noisy_step

        return batch_steps