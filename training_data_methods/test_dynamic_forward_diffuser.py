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
    merged_data_list, merged_data_labels = generator.convert_data_to_training_format()

    print(merged_data_list[0].shape)

def test2(): # Check create_batches function
    generator = DynamicForwardDiffuser()

    training_data_path = "training_data.json"

    generator.load_training_data(training_data_path)
    merged_data_list, merged_data_labels = generator.convert_data_to_training_format()

    merged_data_array = np.stack(merged_data_list)
    print(merged_data_array.shape)

    merged_labels_array = np.array(merged_data_labels)
    print(merged_labels_array.shape)

    batch_size = 8
    batches = generator.create_batches(merged_data_array, 8)
    print(batches[0].shape)

if __name__ == '__main__':
    #test1()
    test2()