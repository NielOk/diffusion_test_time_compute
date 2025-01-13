'''
Class that generates non-noisy training data for the diffusion test time compute problem, which the dynamic forward
diffusion training data generator will use to dynamically generate noisy training data during training.
'''

from PIL import Image
import numpy as np
from typing import Tuple, Dict
import json

class NonNoisyDataGenerator:

    def __init__(self) -> None:
        self.background_color = None
        self.square_color = None
        self.triangle_color = None
        self.image_size = None
        self.square_size = None
        self.triangle_size = None

    def draw_square(self, 
                    image_size: Tuple[int, int],
                    square_size: int, 
                    square_color: Tuple[int, int, int],
                    background_color: Tuple[int, int, int], # (255, 255, 255) is white, (0, 0, 0) is black
                    image_path: str = "",
                    inspect: bool = False
                    ) -> np.ndarray:
        '''
        Draws a square on an image and saves it to a file.
        '''
        # Create a blank image
        canvas_matrix = np.full((image_size[0], image_size[1], 3), background_color, dtype=np.uint8)
        
        # Figure out coordinates and replace
        width, height = image_size
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size
        canvas_matrix[top:bottom, left:right] = square_color

        if inspect and image_path != "":
            image = Image.fromarray(canvas_matrix)
            image.save(image_path)

        # Save parameters to class
        self.background_color = background_color
        self.square_color = square_color
        self.image_size = image_size
        self.square_size = square_size

        return canvas_matrix
    
    def draw_triangle(self, 
                    image_size: Tuple[int, int],
                    triangle_size: Tuple[int, int],
                    triangle_color: Tuple[int, int, int],
                    background_color: Tuple[int, int, int], # (255, 255, 255) is white, (0, 0, 0) is black
                    image_path: str = "",
                    inspect: bool = False
                    ) -> np.ndarray:
        '''
        Draws a triangle on an image and saves it to a file.
        '''
        # Create a blank image
        canvas_matrix = np.full((image_size[0], image_size[1], 3), background_color, dtype=np.uint8)
        
        # Figure out coordinates and draw triangle
        # Triangle dimensions and center
        width, height = image_size
        triangle_width, triangle_height = triangle_size

        center_x, center_y = width // 2, height // 2
        half_width = triangle_width // 2

        # Triangle vertices
        top_vertex = (center_x, center_y - triangle_height // 2)  # Top vertex
        left_vertex = (center_x - half_width, center_y + triangle_height // 2)  # Bottom-left vertex
        right_vertex = (center_x + half_width, center_y + triangle_height // 2)  # Bottom-right vertex

        # Fill the triangle region
        for y in range(height):
            for x in range(width):
                # Check if the point (x, y) is inside the triangle using the barycentric method
                b1 = (x - left_vertex[0]) * (top_vertex[1] - left_vertex[1]) - (y - left_vertex[1]) * (top_vertex[0] - left_vertex[0]) >= 0
                b2 = (x - right_vertex[0]) * (left_vertex[1] - right_vertex[1]) - (y - right_vertex[1]) * (left_vertex[0] - right_vertex[0]) >= 0
                b3 = (x - top_vertex[0]) * (right_vertex[1] - top_vertex[1]) - (y - top_vertex[1]) * (right_vertex[0] - top_vertex[0]) >= 0
                if b1 == b2 == b3:  # Point is inside the triangle
                    canvas_matrix[y, x] = triangle_color
        
        if inspect and image_path != "":
            image = Image.fromarray(canvas_matrix)
            image.save(image_path)

        # Save parameters to class
        self.background_color = background_color
        self.triangle_color = triangle_color
        self.image_size = image_size
        self.triangle_size = triangle_size

        return canvas_matrix
    
    def generate_training_data(self, 
                            num_images: int, 
                            save_path: str,
                            squares: bool = True,
                            triangles: bool = True,
                            background_color: Tuple[int, int, int] = (255, 255, 255), # Default background color for all images is white
                            image_size: Tuple[int, int] = (32, 32) # Default image size is 32x32
                            ) -> None:
        
        '''
        Generates non-noisy data for the diffusion model, to be used for training.
        Image size is 32x32. Square size, color and triangle size, color is different 
        for every image. Matrix values are saved to a json file. 
        '''

        num_squares = 0
        num_triangles = 0

        if squares and triangles:
            num_squares = num_images // 2
            num_triangles = num_images - num_squares
        elif squares:
            num_squares = num_images
        elif triangles:
            num_triangles = num_images
        
        training_data_dict = { # Dictionary to store training data. Major keys are "squares" and "triangles". Each major key has a list of image arrays as values.
            "squares": [],
            "triangles": []
        }
        # Get square possible dimensions
        square_max_length = np.minimum(image_size[0], image_size[1])

        # Generate squares training data
        for i in range(num_squares):
            square_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            square_size = np.random.randint(1, square_max_length) # Give a buffer of 1 pixel for both miniumum and maximum size

            square_matrix = self.draw_square(image_size, square_size, square_color, background_color)

            training_data_dict["squares"].append(square_matrix.tolist())

        # Get triangle possible dimensions
        triangle_min_base_or_height = 6
        triangle_max_base = np.minimum(image_size[0], image_size[1])

        # Generate triangles training data
        for i in range(num_triangles):
            triangle_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            triangle_size = (np.random.randint(triangle_min_base_or_height, triangle_max_base), np.random.randint(triangle_min_base_or_height, triangle_max_base))

            triangle_matrix = self.draw_triangle(image_size, triangle_size, triangle_color, background_color)

            training_data_dict["triangles"].append(triangle_matrix.tolist())

        with open(save_path, "w") as f:
            json.dump(training_data_dict, f, indent=4)

        print(f"Generated non-noisy data for {num_squares} squares and {num_triangles} triangles.")