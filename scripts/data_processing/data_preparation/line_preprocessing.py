import cv2
import numpy as np
from pathlib import Path
from typing import Union
import logging

class LinePreprocessor:
    """
    Handles preprocessing for line-level OCR images.
    Creates images with fixed height (384px) and preserves original text width.
    Text should occupy approximately 278px height (72% of total height).
    """

    def __init__(self, 
                 target_height: int = 384, 
                 text_height_ratio: float = 0.72,
                 max_width: int = 2000,
                 background_color: tuple = (255, 255, 255)):
        """
        Initialize line preprocessor.

        Args: 
            target_height: Fixed height for all line images (384px)
            text_height_ratio: Ratio of height that text should occupy (0.72 = 278px/384px)
            max_width: Maximum width to prevent extremely wide images (2000px)
            background_color: RGB color for padding (white by default)
        """
        self.target_height = target_height
        self.text_height_ratio = text_height_ratio
        self.target_text_height = int(target_height * text_height_ratio)  # ~278px
        self.max_width = max_width
        self.background_color = background_color
        
        logging.info("LinePreprocessor initialized:")
        logging.info(f"  Target height: {target_height}px (fixed)")
        logging.info(f"  Text height: {self.target_text_height}px ({text_height_ratio:.1%})")
        logging.info(f"  Max width: {max_width}px")
        logging.info("  Width: Preserves original proportions")

    def preprocess_line_image(self, 
                             image: Union[np.ndarray, str, Path], 
                             writer_id: str = None, 
                             text: str = None) -> np.ndarray:
        """
        Preprocess a line image for line-level OCR.
        
        Args:
            image: Input image as numpy array or path to image file
            writer_id: Writer ID (for debugging)
            text: Line text (for debugging)
            
        Returns:
            Preprocessed line image with fixed height (384px) and preserved width
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        original_height, original_width = image.shape[:2]
        
        # Scale text to target text height while preserving aspect ratio
        scale_factor = self.target_text_height / original_height if original_height > 0 else 1.0
        scaled_width = int(original_width * scale_factor)
        scaled_height = self.target_text_height
        
        # Check if width becomes too extreme and apply max_width constraint
        if scaled_width > self.max_width:
            # Recalculate using width constraint instead
            width_scale_factor = self.max_width / original_width if original_width > 0 else 1.0
            scaled_width = self.max_width
            scaled_height = int(original_height * width_scale_factor)
            # Ensure scaled height doesn't exceed target height
            if scaled_height > self.target_height - 20:  # Leave some padding
                scaled_height = self.target_height - 20
            scale_factor = width_scale_factor
            
            logging.debug(f"Applied max_width constraint: {original_width}x{original_height} -> {scaled_width}x{scaled_height}")
        else:
            logging.debug(f"Normal proportional scaling: {original_width}x{original_height} -> {scaled_width}x{scaled_height}")
        
        # Resize image to calculated dimensions (preserving proportions or applying constraints)
        if scale_factor != 1.0:
            resized_image = cv2.resize(image, (scaled_width, scaled_height), 
                                     interpolation=cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC)
        else:
            resized_image = image
        
        # Create final canvas with preserved width
        final_width = scaled_width
        if len(image.shape) == 3:
            canvas = np.full((self.target_height, final_width, image.shape[2]), 
                           self.background_color, dtype=image.dtype)
        else:
            canvas = np.full((self.target_height, final_width), 
                           self.background_color[0], dtype=image.dtype)
        
        # Calculate vertical positioning (center the text vertically)
        text_y_offset = (self.target_height - scaled_height) // 2
        
        # Place scaled text on canvas (horizontally: full width, vertically: centered)
        canvas[text_y_offset:text_y_offset + scaled_height, :] = resized_image
        
        # Debug logging
        if text and writer_id:
            logging.debug(f"Line preprocessing - {writer_id}: '{text[:50]}...'")
            logging.debug(f"  Original: {original_width}x{original_height}")
            logging.debug(f"  Scaled text: {scaled_width}x{scaled_height} (factor: {scale_factor:.2f})")
            logging.debug(f"  Final canvas: {final_width}x{self.target_height}")
            logging.debug(f"  Vertical position: {text_y_offset}")
        
        return canvas

    def set_writer_reference(self, writer_id: str, ref_h: int, ref_w: int):
        """
        Compatibility method with ImagePreprocessor interface.
        For line preprocessing, we don't use writer references but need this for compatibility.
        """
        logging.debug(f"LinePreprocessor: Ignoring writer reference for {writer_id} ({ref_w}x{ref_h})")
        pass

    def preprocess_single_image(self, image: Union[np.ndarray, str, Path], writer_id=None, word=None) -> np.ndarray:
        """
        Compatibility method with ImagePreprocessor interface.
        For line-level processing, 'word' parameter contains the full line text.
        """
        return self.preprocess_line_image(image, writer_id, word)