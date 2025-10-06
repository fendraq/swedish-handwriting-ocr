import cv2
import numpy as np
import math

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

def extract_word_region(image: np.ndarray, coords: tuple, margin_px: int = 8) -> np.ndarray:
    """
    Cuts a rectangular region of image with a margin

    Args:
        image: Fullsize image as NumPy array
        coords: (x1, y1, x2, y2) pixel coordinates
        margin_px: set margin from printed lines in rectangle
    Returns:
        Cut word region as NumPy array

    Raises: 
        ValueError: If region is empty
    """

    x1, y1, x2, y2 = coords
    x1 += margin_px
    y1 += margin_px
    x2 -= margin_px
    y2 -= margin_px

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

def detect_rotation_angle(detected_markers: dict, expected_markers: dict) -> float:
    """
    Calculate rotation angle based on the reference markers position

    Args:
        detected_markers: {marker_id: (x, y)} from HoughCircles
        expected_markers: {marker_id: (x, y)} from PDF-metadata

    Returns:
        rotation_angle: angle in degrees (+ = clockwise)
    """

    # Debug
    print(f"DEBUG: Detected markers {list(detected_markers.keys())}")
    print(f"DEBUG: Expected markers {list(expected_markers.keys())}")

    # Test horizontal markers
    if 'top_left' in detected_markers and 'top_right' in detected_markers and 'top_left' in expected_markers and 'top_right' in expected_markers:
        detected_dx = detected_markers['top_right'][0] - detected_markers['top_left'][0]
        detected_dy = detected_markers['top_right'][1] - detected_markers['top_left'][1]
        expected_dx = expected_markers['top_right'][0] - expected_markers['top_left'][0]
        expected_dy = expected_markers['top_right'][1] - expected_markers['top_left'][1]

        angle_detected = math.atan2(detected_dy, detected_dx)
        angle_expected = math.atan2(expected_dy, expected_dx)

        # Calculate correction angle - positive for clockwise rotation
        rotation_angle = math.degrees(angle_detected - angle_expected)

        print(f"DEBUG: Horizontal line rotation: {rotation_angle:.3f}")
        return rotation_angle
    
    elif 'top_left' in detected_markers and 'bottom_left' in detected_markers and 'top_left' in expected_markers and 'bottom_left' in expected_markers:
        detected_dx = detected_markers['bottom_left'][0] - detected_markers['top_left'][0]
        detected_dy = detected_markers['bottom_left'][1] - detected_markers['top_left'][1]
        expected_dx = expected_markers['bottom_left'][0] - expected_markers['top_left'][0]
        expected_dy = expected_markers['bottom_left'][1] - expected_markers['top_left'][1]

        angle_detected = math.atan2(detected_dy, detected_dx)
        angle_expected = math.atan2(expected_dy, expected_dx)

        # Calculate correction angle - positive for clockwise rotation
        rotation_angle = math.degrees(angle_detected - angle_expected)

        print(f"DEBUG: Vertical line rotation: {rotation_angle:.3f}")
        return rotation_angle
    
    return 0.0

def rotate_image_to_correct_orientation(image: np.ndarray, rotation_angle: float, detected_markers: dict) -> tuple:
    """
    Rotate image to correct angle
    
    Args:
        image: OpenCV image array
        rotate_angle: degrees to rotate (+ = clockwise)
        detected_markers: to calculate rotation center

    Returns:
        tuple: (rotated_image, new_markers_dict)
    """
    # Check if rotation is needed
    if abs(rotation_angle) < 0.05:
        return image, detected_markers
    
    height, width = image.shape[:2]

    # Use center of detected markers as rotation center
    if detected_markers:
        center_x = sum(pos[0] for pos in detected_markers.values()) / len(detected_markers)
        center_y = sum(pos[1] for pos in detected_markers.values()) / len(detected_markers)

    else:
        center_x = width / 2
        center_y = height / 2

    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(255, 255, 255))
    
    # Update marker positions
    new_markers = {}
    for marker_id, (x, y) in detected_markers.items():
        point = np.array([[x, y]], dtype=np.float32)
        transformed_point = cv2.transform(point.reshape(1, 1, 2), rotation_matrix)
        new_x, new_y = transformed_point[0, 0]
        new_markers[marker_id] = (int(new_x), int(new_y))

    return rotated_image, new_markers

def auto_correct_image_orientation(image_path: str, detected_markers: dict, expected_markers: dict, 
                                    rotation_threshold: float = 0.3) -> tuple:
    """
    Main function that detects and correcting image rotation

    Args:
        image_path: path to original image
        detected_markers: markers detected in image
        expected_markers: expected markers from PDF
        rotation_threshold: minimum rotation in degrees to make correction

    Returns:
        tuple: (corrected_image_array, corrected_markers_dict, rotation_applied)
    """

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Calculate rotation angle
    rotation_angle = detect_rotation_angle(detected_markers, expected_markers)

    # Check if roation is needed
    if abs(rotation_angle) > rotation_threshold:
        print(f"Detected rotation: {rotation_angle:.2f} degrees, correction...")

        # Rotate image
        corrected_image, corrected_markers = rotate_image_to_correct_orientation(
            image, rotation_angle, detected_markers
        )

        return corrected_image, corrected_markers, rotation_angle
    else:
        print(f"Rotation {rotation_angle:.2f} degrees is within threshold, no correction needed")
        return image, detected_markers, 0.0
