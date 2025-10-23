import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from sklearn.model_selection import train_test_split
import random

logger = logging.getLogger(__name__)

class TrOCRDatasetSplitter:
    """ 
    Creates simple train/val/test splits in TrOCR-compatible format.
    Based on Microsoft TrOCR standard approach from microsoft/unilm repository.
    """
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Set random seeds for reproducible splits
        random.seed(random_state)

        self.annotations: List[Dict[str, Any]] = []
        self.splits: Dict[str, List[Dict[str, Any]]] = {}

    def load_annotations(self, annotations_path: Path) -> None:
        """ Load annotations from JSON file """
        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        logger.info(f"Loaded {len(self.annotations)} annotations from {annotations_path}")

    def create_simple_splits(self) -> None:
        """
        Create simple random splits following TrOCR standard approach.
        Uses sklearn train_test_split for clean 70/15/15 distribution.
        """
        if not self.annotations:
            raise ValueError("No annotations loaded. Call load_annotations() first")
        
        # First split: separate train from (val + test)
        train_data, temp_data = train_test_split(
            self.annotations,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.random_state,
            shuffle=True
        )
        
        # Second split: separate val from test
        if temp_data:
            val_data, test_data = train_test_split(
                temp_data,
                test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
                random_state=self.random_state,
                shuffle=True
            )
        else:
            val_data, test_data = [], []

        self.splits = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        logger.info(f"Created splits: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    def save_trocr_splits(self, output_dir: Path) -> Dict[str, Path]:
        """
        Save train/val/test splits in TrOCR format: gt_train.txt, gt_val.txt, gt_test.txt
        
        Format per line: image_path<TAB>ground_truth_text
        Based on microsoft/unilm TrOCR data.py Receipt53K format.
        
        Returns:
            Dict mapping split names to file paths
        """
        if not self.splits:
            raise ValueError("No splits created. Call create_simple_splits() first")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for split_name, annotations in self.splits.items():
            if not annotations:
                logger.warning(f"Skipping empty {split_name} split")
                continue

            # TrOCR expects gt_<split>.txt format
            txt_file = output_dir / f"gt_{split_name}.txt"

            with open(txt_file, 'w', encoding='utf-8') as f:
                for annotation in annotations:
                    # Format: image_path<TAB>ground_truth_text
                    # Following microsoft/unilm Receipt53K format
                    image_path = annotation['image_path']
                    ground_truth = annotation['ground_truth_text']
                    
                    # Write tab-separated format
                    f.write(f"{image_path}\t{ground_truth}\n")

            saved_files[split_name] = txt_file
            logger.info(f"Saved {len(annotations)} samples to {txt_file}")

        return saved_files

def create_dataset_splits(annotations_path: Path, output_dir: Path, 
                          train_ratio: float = 0.7, val_ratio: float = 0.15,
                          test_ratio: float = 0.15, random_state: int = 42,
                          use_augmented: bool = True) -> Dict[str, Path]:
    """
    Main function to create TrOCR-compatible dataset splits from annotations.
    
    Creates gt_train.txt, gt_val.txt, gt_test.txt files in the format expected
    by Microsoft TrOCR training pipeline.
    
    Args:
        annotations_path: Path to annotations.json file
        output_dir: Directory to save gt_train.txt, gt_val.txt, gt_test.txt
        train_ratio: Proportion for training set (default 0.7)
        val_ratio: Proportion for validation set (default 0.15) 
        test_ratio: Proportion for test set (default 0.15)
        random_state: Random seed for reproducible splits
        use_augmented: If True, uses annotations_augmented,json
        
    Returns:
        Dictionary mapping split names to TXT file paths
    """
    logger.info(f"Creating TrOCR dataset splits from {annotations_path}")

    # Use augmented annotations if requested and available
    if use_augmented:
        augmented_path = annotations_path.parent / "annotations_augmented.json"
        if augmented_path.exists():
            annotations_path = augmented_path
            logger.info(f"Using augmented annotations: {annotations_path}")
        else:
            logger.warning(f"Augmented annotations not found at {augmented_path}, using original")

    # Create splitter instance
    splitter = TrOCRDatasetSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )

    # Load annotations and create splits
    splitter.load_annotations(annotations_path)
    splitter.create_simple_splits()

    # Save as TrOCR-compatible TXT files
    saved_files = splitter.save_trocr_splits(output_dir)

    logger.info(f"TrOCR dataset splitting complete. Files saved to {output_dir}")
    return saved_files