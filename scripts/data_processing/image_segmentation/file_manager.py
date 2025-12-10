import cv2
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime
from config.paths import ensure_dir

# Global counter for date-based segment IDs 
_daily_segment_counter = {}

def get_next_segment_id() -> str:
    """
    Generate next segment ID in format: YYYYMMDD_sl_000
    Counter continues for all segmentations on the same day.
    
    Returns:
        Unique segment ID for today
    """
    today = datetime.now().strftime("%Y%m%d")
    
    if today not in _daily_segment_counter:
        _daily_segment_counter[today] = 0
    else:
        _daily_segment_counter[today] += 1
    
    return f"{today}_sl_{_daily_segment_counter[today]:03d}"

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

def save_word_segment(image: np.ndarray, text: str, category: str, writer_id: str, word_id: str, output_path: str, page_number: int = None) -> str:
    """
    Saves JPG-image with date-based unique ID naming.
    New format: YYYYMMDD_sl_000.jpg (continues counter per day)

    Args:
        image: Word region as NumPy array
        text: Text label of word (not used in filename anymore)
        category: Category (for metadata)
        writer_id: Writer id (for metadata)
        word_id: Unique word-ID from metadata (for metadata)
        output_path: Path to images directory 
        page_number: Page number (for metadata)

    Returns:
        Path to saved image file
    """

    # Generate unique segment ID for today
    segment_id = get_next_segment_id()
    
    images_dir = Path(output_path)
    jpg_path = images_dir / f"{segment_id}.jpg"
    
    success = cv2.imwrite(str(jpg_path), image)
    if not success:
        raise ValueError(f"Failed to save image: {jpg_path}")
    
    # Note: Text content now stored in gt_*.txt files, not filenames
    # Filename is just unique ID: 20251107_sl_000.jpg

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