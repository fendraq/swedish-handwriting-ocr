import cv2
import json
import numpy as np
from typing import List, Tuple
from .reference_detector import ReferenceDetector
from .image_processor import auto_correct_image_orientation

def load_metadata(metadata_path: str) -> dict:
    """ 
    Reads JSON-metadata from PDF-generation 
    
    Args: 
        metadata_path: path to complete_template_metadata.json

    Returns:
        Dict with all meta data froom PDF-generation
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in metadata file: {metadata_path}")


def pdf_to_pixel_coordinates(pdf_coords: List[float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """ 
    Converts PDF-coordinates to pixels

    Args: 
        pdf_coords: [x1, y1, x2, y2] from metadata
        img_width: Width of image in pixels
        img_height: height of image in pixels

    Returns:
        (img_x1, img_y1, img_x2, img_y2) as int
    """
    PDF_WIDTH = 595
    PDF_HEIGHT = 842

    scale_x = img_width / PDF_WIDTH
    scale_y = img_height / PDF_HEIGHT

    x1, y1, x2, y2 = pdf_coords

    img_x1 = x1 * scale_x
    img_x2 = x2 * scale_x

    img_y1 = (PDF_HEIGHT - y2) * scale_y
    img_y2 = (PDF_HEIGHT - y1) * scale_y

    return(int(img_x1), int(img_y1), int(img_x2), int(img_y2))


def validate_coordinates(coords: Tuple[int, int, int, int], img_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Validating coordinates being inside image boundaries

    Args:
        coord: (x1, y1, x2, y2) pixel coordinates
        img_shape: (height, width) from image.shape
    Returns:
        Corrected coordinates which fit image
    """

    x1, y1, x2, y2 = coords
    img_height, img_width = img_shape[:2]

    # Clamp coordinates to image shape
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 +1

    return (x1, y1, x2, y2)

def get_words_for_page(metadata, page_number):
    """
    Filter words belonging to a specific page

    Args:
        metadata: Complete metadata from load_metadata()
        page_number: Which page to filter

    Returns:
        List with word-dictionaries of the page
    """

    words_list = metadata['words']
    page_words = [word for word in words_list if word['page'] == page_number]

    return page_words

def transform_coordinates_with_references(image_path: str, metadata: dict, target_page: int) -> List[List[int]]:
    """
    Transform PDF coordinates to image coordinates using reference markers.

    Args:
        image_path: Path to scanned image
        metadata: Metadata dictionary with reference system and word coordinates
        target_page: Page number to process

    Returns: 
        List of transformed [x1, y1, x2, y2] coordinates
    """

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    page_words = get_words_for_page(metadata, target_page)
    
    # Initiate reference detector
    ref_detector = ReferenceDetector(
        marker_radius_mm=metadata['reference_system']['circle_radius_mm']
    )

    # Detect reference markers in image
    detected_markers = ref_detector.detect_circular_markers(image_path)

    # Auto-correct image rotation if markers are detected
    if len(detected_markers) >= 2:
        try:
            pdf_markers = metadata['reference_system']['reference_markers']
            corrected_image, corrected_markers, rotation_applied = auto_correct_image_orientation(
                image_path, detected_markers, pdf_markers, rotation_threshold=0.3
            )

            # Replace original image with corrected version if rotation was applied
            if abs(rotation_applied) > 1:
                cv2.imwrite(image_path, corrected_image)
                print(f"Applied {rotation_applied:.2f} degrees rotation correction and saved to {image_path}")

                # Use corrected markers for transformation
                detected_markers = corrected_markers
                
                # Reload the corrected image for coordinate transformation
                image = corrected_image

        except Exception as e:
            print(f"Warning: Auto-rotation failed: {e}, continuing with original image")

    if len(detected_markers) < 2:
        print(f"Warning: Only {len(detected_markers)} markers detected. Falling back to simple conversion")
        words_coords = []
        img_h, img_w = image.shape[:2]
        for word in page_words:
            coords = pdf_to_pixel_coordinates(word['position'], img_w, img_h)
            words_coords.append(list(coords))
        return words_coords
    
    pdf_markers = metadata['reference_system']['reference_markers']

    try:
        # Calculate transformation matrix
        transformation_matrix = ref_detector.calculate_transformation(detected_markers, pdf_markers)

        # Transform all word coordinates
        transformed_coords = []
        for word in page_words:
            pdf_coords = word['position']

            pdf_points = np.float32([
                [pdf_coords[0], pdf_coords[1]],  # top-left
                [pdf_coords[2], pdf_coords[1]],  # top-right
                [pdf_coords[2], pdf_coords[3]],  # bottom-right
                [pdf_coords[0], pdf_coords[3]]   # bottom-left
            ])

            # Apply transformation
            if transformation_matrix.shape == (3, 3):
                img_points = cv2.perspectiveTransform(
                    pdf_points.reshape( -1, 1, 2),
                    transformation_matrix
                ).reshape(-1, 2)
            else:
                img_points = cv2.transform(
                    pdf_points.reshape(-1, 1, 2),
                    transformation_matrix
                ).reshape(-1, 2)
            
            # Get bounding box of transformed rectangle
            x_coords = img_points[:, 0]
            y_coords = img_points[:, 1]

            x1, x2 = int(min(x_coords)), int(max(x_coords))
            y1, y2 = int(min(y_coords)), int(max(y_coords))

            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            transformed_coords.append([x1, y1, x2, y2])

        print(f"Successfully transformed {len(transformed_coords)} word coordinates using reference markers")
        return transformed_coords
    
    except Exception as e:
        print(f"Error in reference-based transformation: {e}")
        print("Falling back to simple coordinate conversion")
        words_coords = []
        img_h, img_w = image.shape[:2]
        for word in page_words:
            coords = pdf_to_pixel_coordinates(word['position'], img_w, img_h)
            words_coords.append(list(coords))
        return words_coords
    

def visualize_transformation(image_path: str, metadata: dict, output_path: str = None, target_page: int = None) -> np.ndarray:
    """
    Visualize detected reference markers and transformed coordinates.

    Args:
        image_path: Path to scanned image
        metadata: Metadata dictionary
        output_path: Optional path to save visualization
        target_page: Page number to visualize (1-indexed), None for all pages

    Returns:
        Image with visualizations
    """

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image {image_path}")

    # Detect and visualize reference markers
    ref_detector = ReferenceDetector(
        marker_radius_mm=metadata['reference_system']['circle_radius_mm']
    )

    detected_markers = ref_detector.detect_circular_markers(image_path)
    vis_image = ref_detector.visualize_detected_markers(image, detected_markers)
    
    # Get transformed coordinates
    try:
        transformed_coords = transform_coordinates_with_references(image_path, metadata, target_page)

        # Draw word regions
        for i, coords in enumerate(transformed_coords):
            x1, y1, x2, y2 = coords
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis_image, f"{i}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        print(f"Drew {len(transformed_coords)} word regions")

    except Exception as e: 
        print(f"Could not transform coordinates: {e}")

    if output_path:
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved on {output_path}")

    return vis_image

def get_coordinates_with_fallback(image_path: str, metadata: dict, use_references: bool = True, target_page: int = None) -> List[List[int]]:
    """
    Get coordinates using reference markers with fallback to simple conversion.

    Args:
        image_path: Path to scanned image
        metadata: Metadata dictionary
        use_references: Wether to try reference marker detection first
        target_page: Page number to filter words for (1-indexed)

    Returns:
        List of [x1, y1, x2, y2] coordinates
    """

    if use_references and 'reference_system' in metadata:
        try:
            return transform_coordinates_with_references(image_path, metadata, target_page)
        except Exception as e:
            print(f"Reference-based transformation failed: {e}")
            print("Falling back to simple coordinate conversion")

    # Fallback to simple convesion
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Filter words for target page
    page_words = get_words_for_page(metadata, target_page)
    
    words_coords = []
    img_h, img_w = image.shape[:2]
    for word in page_words:
        coords = pdf_to_pixel_coordinates(word['position'], img_w, img_h)
        words_coords.append(list(coords))
    return words_coords