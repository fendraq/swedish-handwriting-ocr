import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Union
import os
import logging

class ImagePreprocessor:
    """
    Handles standardized image preproceessing for TrOCR training.
    Resizes images to 384x384 with bottom-left positioning.
    """

    def __init__(self, target_size: int = 384, background_color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Initiate preprocessor.

        Args: 
            target_size: Target square size for TrOCR (384x384)
            background_color: RGB color for padding (white by default)
        """

        self.target_size =  target_size
        self.background_color = background_color

    def is_within_target_size(self, image: np.ndarray) -> bool:
        """
        Check if image dimensions are within target size.

        Args: 
            Input image (H, W) or (H, W, C)

        Returns: True if both width and height are <= target_size
        """

        height, width = image.shape[:2]
        return width <= self.target_size and height <= self.target_size
    
    def resize_large_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image that exceeds target size while maintaining aspect ratio.

        Args:
            image: Input image (H, W) or (H, W, C)

        Returns:
            Resized image that fits within target_size
        """

        height, width = image.shape[:2]

        # Calculate scaling factor to fit within target size
        scale_factor = min(
            self.target_size / width,
            self.target_size / height
        )

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                          
    def add_top_right_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Add top and right padding to position image at bottom-left of 384x384 canvas.

        Args:
            image: Input image (H, W) or (H, W, C)

        Returns: 384x384 image with input positioned at bottom-left
        """
        height, width = image.shape[:2]

        # Create white canvas
        if len(image.shape) == 3:
            canvas = np.full((self.target_size, self.target_size, image.shape[2]), 
                             self.background_color, dtype=image.dtype)
        else:
            canvas = np.full((self.target_size, self.target_size), 
                             self.background_color[0], dtype=image.dtype)
        
        # Position for bottom-left placement
        start_y = self.target_size - height
        start_x = 0

        # Place image on canvas
        canvas[start_y:start_y + height, start_x:start_x + width] = image

        return canvas
    
    def preprocess_single_image(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Complete preprocessing pipeline for single image.

        Args:
            image: Input image as numpy array or path to image file

        Returns: Preprocessed 384x384 image
        """

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
            
        # Hybrid approach: check size first
        if not self.is_within_target_size(image):
            image = self.resize_large_image(image)

        final_image = self.add_top_right_padding(image)

        return final_image

