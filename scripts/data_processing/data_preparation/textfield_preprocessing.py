import cv2
import numpy as np
from pathlib import Path
from typing import Union
import logging

class TextFieldPreprocessor:
    """
    Minimal preprocessing for text-field images destined for YOLO processing.
    
    Philosophy: Let YOLO handle all preprocessing automatically.
    YOLO was trained on original document images and handles scaling/padding internally.
    """

    def __init__(self):
        """Initialize minimal text field preprocessor."""
        logging.info("TextFieldPreprocessor initialized:")
        logging.info("  Mode: Minimal (preserves original images)")
        logging.info("  YOLO handles: Scaling, padding, normalization")

    def preprocess_textfield_image(self, 
                                  image: Union[np.ndarray, str, Path], 
                                  writer_id: str = None, 
                                  text: str = None) -> np.ndarray:
        """
        Minimal preprocessing - just load and return original image.
        
        Args:
            image: Input image as numpy array or path to image file
            writer_id: Writer ID (for debugging)
            text: Text content description (for debugging)
            
        Returns:
            Original image unchanged (YOLO handles all preprocessing)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        
        # Debug logging
        if text and writer_id:
            height, width = image.shape[:2]
            logging.debug(f"TextField minimal processing - {writer_id}: '{text[:50]}...'")
            logging.debug(f"  Preserved original: {width}x{height} (YOLO will handle preprocessing)")
        
        # Return unchanged - YOLO handles everything
        return image

    def set_writer_reference(self, writer_id: str, ref_h: int, ref_w: int):
        """
        Compatibility method with ImagePreprocessor interface.
        Not used in minimal preprocessing.
        """
        pass

    def preprocess_single_image(self, image: Union[np.ndarray, str, Path], writer_id=None, word=None) -> np.ndarray:
        """
        Compatibility method with ImagePreprocessor interface.
        """
        return self.preprocess_textfield_image(image, writer_id, word)