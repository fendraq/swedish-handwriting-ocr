import cv2
import numpy as np

def load_scanned_image(image_path: str) -> np.ndarray:
    """
    Reads scanned image with OpenCV

    Args:
        image_path: Path to JPG-image

    Returns: NumPy array with image data

    Raises:
        VauleError: If image can't be read
    """
    image = cv2.imread(image_path)
    # If RGB is needed later: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    return image

def extract_word_region(image: np.ndarray, coords: tuple) -> np.ndarray:
    """
    Cuts a rectangular region of image

    Args:
        image: Fullsize image as NumPy array
        coords: (x1, y1, x2, y2) pixel coordinates
    Returns:
        Cut word region as NumPy array

    Raises: 
        ValueError: If region is empty
    """

    x1, y1, x2, y2 = coords
    word_image = image[y1:y2, x1:x2]

    if word_image.size == 0:
        raise ValueError(f"Empty word region with coords {coords}")
    
    return word_image


def enhance_word_image(word_image: np.ndarray) -> np.ndarray:
    """
    Enhancing image quality for better OCR

    Args:
        word_image: Word region as NumPy array

    Returns: 
        Enhanced image
    """
    # 2. Senare :
    #    - Kontrast: cv2.convertScaleAbs(word_image, alpha=1.2, beta=10)
    #    - Skärpa: cv2.filter2D() med kernel
    #    - Brusreducering: cv2.fastNlMeansDenoising()
    return word_image

