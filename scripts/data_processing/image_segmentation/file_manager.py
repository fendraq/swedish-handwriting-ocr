import cv2
import json
from pathlib import Path
from typing import Dict, List
import re
import numpy as np
from config.paths import ensure_dir

def create_output_structure(base_path: str, writer_id: str = None, categories: List[str] = None) -> str:
    """
    Creates flat output structure for TrOCR-ready images.

    Args:
        base_path: Base output directory ( trocr_ready_data/vX)
        writer_id: Not used in flat strucure (Keep for compatibiliity)
        categories: Not used in flat strucure (Keep for compatibiliity)

    Returns:
        Path to images directory
    """
    base = Path(base_path)
    images_dir = base / 'images'
    ensure_dir(images_dir)

    return str(images_dir)

def clean_filename(text: str) -> str:
    """
    Clean string for us in file name
    
    Args:
        text: Original text from metadata
        
    Returns:
        Secure text for file name
    """
    clean_text = re.sub(r'[/\\:*?"<>|]', '_', text)
    
    clean_text = clean_text.replace(' ', '_')
    
    clean_text = re.sub(r'_+', '_', clean_text)
    
    clean_text = clean_text.strip('_')
    
    if len(clean_text) > 50:
        clean_text = clean_text[:50].rstrip('_')
    
    if not clean_text:
        clean_text = "unknown"
    
    return clean_text

def save_word_segment(image: np.ndarray, text: str, category: str, writer_id: str, word_id: str, output_path: str, page_number: int = None) -> str:
    """
    Saves JPG-image + TXT-label in flat structure with extended naming.

    Args:
        image: Word region as NumPy array
        text: Text label of word
        category: Category (saved in txt but not filename)
        writer_id: Writer id
        word_id: Unique word-ID from Meta data
        output_path: Path to images directory 
        page_number: Page number for filename

    Returns:
        Path to saved image file
    """

    clean_text = clean_filename(text)

    if page_number is not None:
        base_filename = f"{writer_id}_page{page_number:02d}_{word_id}_{clean_text}"
    else: 
        base_filename = f"{writer_id}_{word_id}_{clean_text}"
    
    images_dir = Path(output_path)

    jpg_path = images_dir / f"{base_filename}.jpg"
    success = cv2.imwrite(str(jpg_path), image)
    if not success:
        raise ValueError(f"Failed to save image: {jpg_path}")
    
    txt_path = images_dir / f"{base_filename}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)

    return str(jpg_path)

def generate_segmentation_report(results: Dict[str, List[str]], output_dir: str) -> None:
    """
    Creates a conclusion of segmentation results

    Args:
        results: Dict with source_image -> [list of segmented filese]
        output_dir: Folder for report save
    """
    total_words = sum(len(word_list) for word_list in results.values())

    summary = {
        "total_source_images": len(results),
        "total_segmented_words": total_words,
        "average_words_per_image": total_words / len(results) if results else 0,
        "source_files": list(results.keys())
    }

    report_path = Path(output_dir) / "segmentation_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== SEGMENTATION REPORT ===")
    print(f"Total source images: {summary['total_source_images']}")
    print(f"Total segmented words: {summary['total_segmented_words']}")
    print(f"Average words per image: {summary['average_words_per_image']:.1f}")
    print(f"\nReport saved to: {report_path}")