import os
import random
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

def main():
    # Parameters: adjust as needed
    sample_count = 500  # number of images to sample
    mnist_root = './MNIST'  # directory where MNIST is stored
    output_folder = './mnist_sample_images'  # directory to save the sampled MNIST images
    noise_output_folder = './noise_sample_images'  # directory to save the noise images

    # Create the output directories if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(noise_output_folder, exist_ok=True)

    # Load the MNIST dataset (assumes images are stored locally)
    dataset = datasets.MNIST(root=mnist_root, train=True, download=True)
    
    # Randomly sample indices from the dataset
    indices = random.sample(range(len(dataset)), sample_count)

    for idx in indices:
        # Process MNIST image
        image, label = dataset[idx]
        # Ensure the image is 28x28 single channel (grayscale)
        image = image.convert('L')
        if image.size != (28, 28):
            image = image.resize((28, 28))
        # Save the MNIST image
        filename = os.path.join(output_folder, f'mnist_{idx}_label_{label}.png')
        image.save(filename)
        print(f"Saved MNIST image: {filename}")

        # Generate a 28x28 random noise image (grayscale)
        noise_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        noise_image = Image.fromarray(noise_array, mode='L')
        # Save the noise image
        noise_filename = os.path.join(noise_output_folder, f'noise_{idx}.png')
        noise_image.save(noise_filename)
        print(f"Saved noise image: {noise_filename}")

if __name__ == '__main__':
    main()
