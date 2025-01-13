'''
Scripts that test non_noisy_data_generator.py
'''

from PIL import Image
import numpy as np

from non_noisy_data_generator import NonNoisyDataGenerator

def test1(): # Test draw_square and draw_triangle functions
    generator = NonNoisyDataGenerator()
    background_color = (255, 255, 255)

    # Draw square
    square_color = (0, 0, 0)
    image_size = (32, 32)
    square_size = 1
    square_path = "square.png"

    # Draw triangle
    triangle_color = (0, 0, 0)
    triangle_size = (6, 6)
    triangle_path = "triangle.png"

    square_matrix = generator.draw_square(image_size, square_size, square_color, background_color, square_path, inspect=True)
    triangle_matrix = generator.draw_triangle(image_size, triangle_size, triangle_color, background_color, triangle_path, inspect=True)

    print(f"Square matrix: {square_matrix}")
    print(f"Triangle matrix: {triangle_matrix}")

if __name__ == '__main__':
    test1()