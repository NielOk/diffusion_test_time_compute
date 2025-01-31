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
    batch_steps, prev_step_noises = generator.batch_uniform_scaled_forward_diffusion(batches[0], num_forward_diffusion_steps)

    # Draw the steps
    for step, batch in batch_steps.items():
        first_row = batch[0]
        image = Image.fromarray(first_row)
        image.show()

def test4(): # Check batch_uniform_beta_schedule_forward_diffusion function
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

    T = 20
    
    # define beta
    T = 20
    beta = np.linspace(0.0001, 0.02, T)  # Uniform beta schedule

    batch_steps, prev_step_noises = generator.batch_beta_schedule_forward_diffusion(batches[0], T, beta)

    # Draw the steps
    for step, batch in batch_steps.items():
        first_row = batch[0]
        image = Image.fromarray(first_row)
        image.show()

def test5(): # temporal embeddings, label embeddings and concatenation test
    t = 1
    shape = (8, 32, 32, 3)
    pe = DynamicForwardDiffuser.sinusoidal_positional_embedding(t, shape)
    print(pe.shape)
    print(pe)

    generator = DynamicForwardDiffuser()
    training_data_path = "training_data.json"

    generator.load_training_data(training_data_path)
    merged_data_list, merged_data_labels = generator.convert_data_to_neural_net_format()

    merged_data_array = np.stack(merged_data_list)
    print(merged_data_array.shape)

    merged_labels_array = np.array(merged_data_labels)
    print(merged_labels_array.shape)

    batch_size = 8
    data_batches = generator.create_batches(merged_data_array, 8)
    label_batches = generator.create_batches(merged_labels_array, 8)
    print(data_batches[0].shape)

    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    le = DynamicForwardDiffuser.label_embedding(labels, shape)
    print(le.shape)
    print(le)

    concatenated_batch_0 = np.concatenate([data_batches[0], pe], axis=-1)
    concatenated_batch_0 = np.concatenate([concatenated_batch_0, le], axis=-1)

    print(concatenated_batch_0.shape)
    print(concatenated_batch_0)
    print(concatenated_batch_0[0, 0, 0, -1])

if __name__ == '__main__':
    #test1()
    #test2()
    #test3()
    test4()
    #test5()