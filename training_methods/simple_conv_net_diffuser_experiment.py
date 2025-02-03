import os
import numpy as np
import torch

from reverse_diffusion_training import SimpleConvNetDiffuser, load_non_noisy_data, cosine_beta_schedule,train_model, test_model, visualize_model_output

TRAINING_METHODS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TRAINING_METHODS_DIR)
TRAINING_DATA_METHODS_DIR = os.path.join(REPO_DIR, 'training_data_methods')

def simple_conv_net_diffuser_experiment():
    # Load training data
    training_data_path = os.path.join(TRAINING_DATA_METHODS_DIR, 'training_data.json')
    X_train_batches, X_test_batches, y_train_batches, y_test_batches, generator = load_non_noisy_data(training_data_path)

    # Define beta
    T = 50
    betas = cosine_beta_schedule(T, s=0.008)

    # Train model
    model_save_filename = 'simple_conv_net_diffuser.pth'
    model = SimpleConvNetDiffuser()
    #train_model(model, X_train_batches, y_train_batches, generator, betas, num_diffusion_steps=T, num_epochs=2, learning_rate=0.001, model_save_filename=model_save_filename)

    # Load trained model 
    trained_model = SimpleConvNetDiffuser()
    model_save_dir = os.path.join(TRAINING_METHODS_DIR, model_save_filename)
    trained_model.load_state_dict(torch.load(model_save_dir))

    # Test model
    #test_model(trained_model, X_test_batches, y_test_batches, generator, betas, num_diffusion_steps=T)

    # Visualize model output
    visualize_model_output(trained_model, betas, num_diffusion_steps=T)

if __name__ == '__main__':
    simple_conv_net_diffuser_experiment()