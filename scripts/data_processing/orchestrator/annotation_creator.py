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
    """ Create annotations from segmented images and metadata """
    def __init__(self, images_dir: Path, metadata_dir: Path):
        self.images_dir = Path(images_dir)
        self.metadata_dir = Path(metadata_dir)
        self.annotations: List[WordAnnotation] = []

    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        Parse segmented image filename to extract metadata
        Expected format: {writer_id}_{page}_{word_id}_{text}
        """
        stem = Path(filename).stem
        parts = stem.split('_')

        if len(parts) < 4:
            raise ValueError(f"Invalid filename format: {filename}")

        writer_id = parts[0]
        page_number = int(parts[1])
        word_id = int(parts[2])
        text = '_'.join(parts[3:])

        return {
            'writer_id': writer_id,
            'page_number': page_number,
            'word_id': word_id,
            'text': text
        }
    
    def create_annotations_from_images(self) -> None:
        """
        Scan images directory and create annotations for all segmented images
        """
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        image_files = list(self.images_dir.glob('*.jpg'))
        logger.info(f"Found {len(image_files)} image files to process")

        for image_file in image_files:
            try:
                file_info = self.parse_filename(image_file.name)

                annotation = WordAnnotation(
                    image_path=str(image_file.relative_to(self.images_dir.parent)),
                    ground_truth_text=file_info['text'],
                    writer_id=file_info['writer_id'],
                    page_number=file_info['page_number'],
                    word_id=file_info['word_id'],
                    category="word",  
                    confidence=1.0  # Manual annotations = 100% confidence
                )
                self.annotations.append(annotation)

            except Exception as e:
                logger.warning(f"Failed to process {image_file.name}: {e}")
                continue

        logger.info(f"Created {len(self.annotations)} annotations")

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
    
def create_annotations_for_version(images_dir: Path, output_dir: Path, metadata_dir: Path = None) -> Path:
    """
    Main function to create annotations for a version directory

    Args:
        images_dir: Path to directory containing segmented images
        output_dir: Path where annotations.json will be saved
        metadata_dir: Optional path to metadata directoory (for future expansion)

    Returns:
        Path to create annotations.json fil
    """
    logger.info(f"Creating annotations for images in: {images_dir}")

    creator = AnnotationCreator(images_dir, metadata_dir or images_dir.parent)

    creator.create_annotations_from_images()

    annotations_file = output_dir / 'annotations.json'
    creator.save_annotations(annotations_file)

    stats = creator.get_annotation_stats()
    logger.info(f"Annotations statistics: {stats}")

    return annotations_file
    
if __name__ == '__main__':
    # Example usage
    images_dir = Path('trocr_ready_data/v1/images')
    output_dir = Path('trocr_ready_data/v1')

    annotations_file = create_annotations_for_version(images_dir, output_dir)
    print(f"Created annotations: {annotations_file}")
            