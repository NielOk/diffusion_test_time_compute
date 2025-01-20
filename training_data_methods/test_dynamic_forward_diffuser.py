'''
Tests for dynamic forward diffuser
'''

import numpy as np
from PIL import Image

from dynamic_forward_diffuser import DynamicForwardDiffuser

def test1(): # Check shape of images after loading training data
    generator = DynamicForwardDiffuser()

    training_data_path = "training_data.json"

    generator.load_training_data(training_data_path)
    merged_data_list, merged_data_labels = generator.convert_data_to_neural_net_format()

    print(merged_data_list[0].shape)

def test2(): # Check create_batches function
    generator = DynamicForwardDiffuser()

    training_data_path = "training_data.json"

    generator.load_training_data(training_data_path)
    merged_data_list, merged_data_labels = generator.convert_data_to_neural_net_format()

    merged_data_array = np.stack(merged_data_list)
    print(merged_data_array.shape)

    merged_labels_array = np.array(merged_data_labels)
    print(merged_labels_array.shape)

    batch_size = 8
    batches = generator.create_batches(merged_data_array, 8)
    print(batches[0].shape)

def test3(): # Check batch_uniform_scaled_forward_diffusion function
    generator = DynamicForwardDiffuser()

    training_data_path = "training_data.json"

    generator.load_training_data(training_data_path)
    merged_data_list, merged_data_labels = generator.convert_data_to_neural_net_format()

    merged_data_array = np.stack(merged_data_list)
    print(merged_data_array.shape)

    merged_labels_array = np.array(merged_data_labels)
    print(merged_labels_array.shape)

    batch_size = 8
    batches = generator.create_batches(merged_data_array, 8)
    print(batches[0].shape)

    num_forward_diffusion_steps = 20
    batch_steps = generator.batch_uniform_scaled_forward_diffusion(batches[0], num_forward_diffusion_steps)

    # Draw the steps
    for step, batch in batch_steps.items():
        first_row = batch[0]
        image = Image.fromarray(first_row)
        image.show()

if __name__ == '__main__':
    #test1()
    #test2()
    test3()