import json
from typing import List, Tuple

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