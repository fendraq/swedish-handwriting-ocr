import cv2
import json
from pathlib import Path
from typing import Dict, List
import re
import numpy as np

def create_output_structure(base_path: str, writer_id: str, categories: List[str]) -> Dict[str, str]:
    """
    Creates file structure of categories

    Args:
        base_path: Folder for segmented images
        writer_id: Identification of the writer (e.g. "writer_001")

    Returns: Dictionary with categories -> path mapping
    """
    base = Path(base_path)

    writer_dir = base / writer_id
    writer_dir.mkdir(parents=True, exist_ok=True)

    category_paths = {}
    for category in categories:
        category_dir = writer_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        category_paths[category] = str(category_dir)

    return category_paths

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

def save_word_segment(image: np.ndarray, text: str, category: str, writer_id: str, word_id:str, output_paths: Dict[str, str]) -> str:
    """
    Saves JPG-image + TXT-label.

    Args:
        image: Word region as NumPy array
        text: Text label of word
        category: Category
        writer_id: Writer id
        word_id: Unique word-ID from Meta data
        output_paths: Path from create_output_structure()

    Returns:
        Path to saved image file
    """

    clean_text = clean_filename(text)

    base_filename = f"{writer_id}_{word_id}_{clean_text}"

    category_dir = Path(output_paths[category])

    jpg_path = category_dir / f"{base_filename}.jpg"
    success = cv2.imwrite(str(jpg_path), image)
    if not success:
        raise ValueError(f"Failed to save image: {jpg_path}")

    txt_path = category_dir / f"{base_filename}.txt"
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

    category_counts = {}
    for word_list in results.values():
        for word_file in word_list:
            word_path = Path(word_file)
            category = word_path.parent.name
            category_counts[category] = category_counts.get(category, 0) +1

    summary = {
        "total_source_images": len(results),
        "total_segmented_words": total_words,
        "words_per_category": category_counts,
        "average_words_per_image": total_words / len(results) if results else 0,
        "source_files": list(results.keys())
    }

    report_path = Path(output_dir) / "segmentation_summary.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== SEGMENTATION REPORT ===")
    print(f"Total source images: {summary['total_source_images']}")
    print(f"Total segmented words: {summary['total_segmented_words']}")
    print(f"Average words per image: {summary['average_words_per_image']:.1f}")
    print(f"\nWords per category:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    print(f"\nReport saved to: {report_path}")