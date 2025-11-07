import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class WordAnnotation:
    """ Represents a single word annotation for TrOCR training """
    image_path: str
    ground_truth_text: str
    writer_id: str
    page_number: int
    word_id: int
    category: str
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """ Convert to dictionary for Json serialization """
        return{
            "image_path": self.image_path,
            "ground_truth_text": self.ground_truth_text,
            "writer_id": self.writer_id,
            "page_number": self.page_number,
            "word_id": self.word_id,
            "category": self.category,
            "confidence": self.confidence
        }
    
class AnnotationCreator:
    """ Create annotations from segmented images and ground truth files """
    def __init__(self, images_dir: Path, gt_file_path: Path):
        self.images_dir = Path(images_dir)
        self.gt_file_path = Path(gt_file_path)
        self.annotations: List[WordAnnotation] = []

    def parse_date_filename(self, filename: str) -> Dict[str, Any]:
        """
        Parse new date-based image filename to extract basic metadata
        Expected format: YYYYMMDD_sl_000.jpg
        Example: 20251107_sl_048.jpg
        """
        stem = Path(filename).stem
        parts = stem.split('_')

        if len(parts) != 3 or parts[1] != 'sl':
            raise ValueError(f"Invalid date-based filename format: {filename}")

        date_str = parts[0]
        sequence_id = int(parts[2])

        return {
            'date': date_str,
            'sequence_id': sequence_id,
            'category': 'word'  # All new segments are single-line words
        }

    def load_ground_truth_mapping(self) -> Dict[str, str]:
        """
        Load ground truth mapping from gt_*.txt file
        Format: image_path<TAB>ground_truth_text
        """
        if not self.gt_file_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.gt_file_path}")
        
        gt_mapping = {}
        with open(self.gt_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                
                parts = line.split('\t')
                if len(parts) != 2:
                    logger.warning(f"Invalid line {line_num} in {self.gt_file_path}: {line}")
                    continue
                
                image_path, ground_truth = parts
                gt_mapping[Path(image_path).name] = ground_truth
        
        logger.info(f"Loaded {len(gt_mapping)} ground truth entries from {self.gt_file_path}")
        return gt_mapping
    
    def create_annotations_from_images(self) -> None:
        """
        Scan images directory and create annotations for all segmented images
        using ground truth mapping from gt file
        """
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Load ground truth mapping
        gt_mapping = self.load_ground_truth_mapping()
        
        image_files = list(self.images_dir.glob('*.jpg'))
        logger.info(f"Found {len(image_files)} image files to process")

        matched_files = 0
        for image_file in image_files:
            try:
                # Check if we have ground truth for this image
                if image_file.name not in gt_mapping:
                    logger.debug(f"No ground truth found for {image_file.name}")
                    continue
                
                ground_truth_text = gt_mapping[image_file.name]
                
                # Parse the date-based filename for metadata
                file_info = self.parse_date_filename(image_file.name)

                annotation = WordAnnotation(
                    image_path=str(image_file.relative_to(self.images_dir.parent)),
                    ground_truth_text=ground_truth_text,
                    writer_id=f"date_{file_info['date']}",  # Use date as writer_id for now
                    page_number=1,  # Default page number for date-based segments
                    word_id=file_info['sequence_id'],
                    category=file_info['category'],
                    confidence=1.0  # Manual annotations = 100% confidence
                )
                self.annotations.append(annotation)
                matched_files += 1

            except Exception as e:
                logger.warning(f"Failed to process {image_file.name}: {e}")
                continue

        logger.info(f"Created {len(self.annotations)} annotations from {matched_files} matched files")

    def save_annotations(self, output_path: Path) -> None:
        """ Save all annotations to JSON file """
        if not self.annotations:
            logger.warning("No annotations to save")
            return
        
        # Convert annotations to dictionaries
        annotations_data = [annotation.to_dict() for annotation in self.annotations]

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotations_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.annotations)} annotaations to {output_path}")

    def get_annotation_stats(self) -> Dict[str, Any]:
        """ Get statistics about annotations """
        if not self.annotations:
            return {'total': 0}

        writers = set(ann.writer_id for ann in self.annotations)
        categories = set(ann.category for ann in self.annotations)

        return {
            "total_annotations": len(self.annotations),
            "unique_writers": len(writers),
            "writers": sorted(list(writers)),
            "categories": sorted(list(categories)),
            "avg_text_length": sum(len(ann.ground_truth_text) for ann in self.annotations) / len(self.annotations)
        }
    
def create_annotations_for_version(images_dir: Path, output_dir: Path, gt_file_path: Path) -> Path:
    """
    Main function to create annotations for a version directory

    Args:
        images_dir: Path to directory containing segmented images
        output_dir: Path where annotations.json will be saved
        gt_file_path: Path to ground truth file (gt_*.txt)

    Returns:
        Path to created annotations.json file
    """
    logger.info(f"Creating annotations for images in: {images_dir}")
    logger.info(f"Using ground truth file: {gt_file_path}")

    creator = AnnotationCreator(images_dir, gt_file_path)

    creator.create_annotations_from_images()

    annotations_file = output_dir / 'annotations.json'
    creator.save_annotations(annotations_file)

    stats = creator.get_annotation_stats()
    logger.info(f"Annotations statistics: {stats}")

    return annotations_file
    
if __name__ == '__main__':
    # Example usage
    images_dir = Path('dataset/trocr_ready_data/v1/images')
    output_dir = Path('dataset/trocr_ready_data/v1')
    gt_file_path = Path('dataset/trocr_ready_data/v1/gt_train.txt')

    annotations_file = create_annotations_for_version(images_dir, output_dir, gt_file_path)
    print(f"Created annotations: {annotations_file}")
            