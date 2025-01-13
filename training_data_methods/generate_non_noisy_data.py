'''
Use the non-noisy data generator to generate and save non-noisy training data for the diffusion test time compute problem.
'''

import json
import numpy as np
from PIL import Image

from non_noisy_data_generator import NonNoisyDataGenerator

if __name__ == '__main__':
    generator = NonNoisyDataGenerator()
    num_images = 1000
    save_path = "training_data.json"

    generator.generate_training_data(num_images, save_path)