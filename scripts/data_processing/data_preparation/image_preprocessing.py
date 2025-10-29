import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Union
import os
import logging
from config.paths import DatasetPaths, ensure_dir

class ImagePreprocessor:
    """
    Handles standardized image preproceessing for TrOCR training.
    Resizes images to 384x384 with bottom-left positioning.
    """

    def __init__(self, target_size: int = 384, background_color: Tuple[int, int, int] = (255, 255, 255),
                 fill_ratio: float = 0.98, max_rel_scale: float = 1.3, min_rel_scale: float = 0.7):
        """
        Initiate preprocessor.

        Args: 
            target_size: Target square size for TrOCR (384x384)
            background_color: RGB color for padding (white by default)
            fill_ratio: How much of the 384x384 that the word can occupy
            max_rel_scale: Max allowed enhancement relative to reference word
            min_rel_scale: Min allowed shrinkage relative to reference word
        """

        self.target_size =  target_size
        self.background_color = background_color
        self.fill_ratio = fill_ratio
        self.max_rel_scale = max_rel_scale
        self.min_rel_scale = min_rel_scale
        self.writer_reference = {}

    def set_writer_reference(self, writer_id, ref_h, ref_w):
        """ Setting scale compared to largest word of writer """
        ref_scale = min(self.fill_ratio * self.target_size / ref_h, self.fill_ratio * self.target_size / ref_w)
        self.writer_reference[writer_id] = (ref_h, ref_w, ref_scale)

    def resize_with_writer_limit(self, image, writer_id, word):
        h, w = image.shape[:2]
        if writer_id not in self.writer_reference:
            print(f"[Warning] No reference for writer {writer_id}, using default scaling")
            scale = min(self.fill_ratio * self.target_size / h, self.fill_ratio * self.target_size / w)
        else:
            ref_h, ref_w, ref_scale = self.writer_reference[writer_id]
            scale = min(self.fill_ratio * self.target_size / h, self.fill_ratio * self.target_size / w)
            max_scale = ref_scale * self.max_rel_scale
            min_scale = ref_scale * self.min_rel_scale
            scale = max(min(scale, max_scale), min_scale)
            print(f"[DEBUG] {writer_id} '{word}': scale={scale:.2f}, ref_scale={ref_scale:.2f}, h={h}, w={w}")
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
                          
    def add_centered_padding(self, image: np.ndarray) -> np.ndarray:
        """
        Add padding around centered word of 384x384 canvas.

        Args:
            image: Input image (H, W) or (H, W, C)

        Returns: 384x384 image with input positioned in center
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
        start_y = (self.target_size - height) // 2
        start_x = (self.target_size - width) // 2

        # Place image on canvas
        canvas[start_y:start_y + height, start_x:start_x + width] = image

        return canvas
    
    def preprocess_single_image(self, image: Union[np.ndarray, str, Path], writer_id=None, word=None) -> np.ndarray:
        """
        Complete preprocessing pipeline for single image.

        Args:
            image: Input image as numpy array or path to image file
            writer_id: Writer-ID (for reference scaling)
            word: Word (for debugging)

        Returns: Preprocessed 384x384 image
        """

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
            
        # Hybrid approach: check size first
        if not self.is_within_target_size(image):
            image = self.resize_with_writer_limit(image, writer_id, word)

        final_image = self.add_centered_padding(image)

        return final_image

