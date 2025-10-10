import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict, Counter
import random
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataSplitter:
    """ Creates stratified train/val/test splits for TrOCR training """
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, random_state: int = 42):
        if abs(train_ratio + val_ratio + test_ratio -1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        random.seed(random_state)

        self.annotations: List[Dict[str, Any]] = []
        self.splits: Dict[str, List[Dict[str, Any]]] = {}

    def load_annotations(self, annotations_path: Path) -> None:
        """ Load annotations from JSON file """
        with open(annotations_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        logger.info(f"Loaded {len(self.annotations)} annotations from {annotations_path}")

    def create_stratified_splits(self) -> None:
        """
        Create stratified splits ensuring:
        1. All unique words appear in training set
        2. Writers are balanced across splits
        3. No data leakage between splits
        """

        if not self.annotations:
            raise ValueError("No annotations loaded. Call load_annotations() first")
        
        # Group by unique words (ground truth text)
        word_groups = defaultdict(list)
        for ann in self.annotations:
            word_groups[ann['ground_truth_text']].append(ann)

        # Separate unique words (only one instance) from common words
        unique_words = []
        common_words = []

        for word, annotations in word_groups.items():
            if len(annotations) == 1:
                unique_words.extend(annotations)
            else:
                common_words.extend(annotations)

        logger.info(f"Found {len(unique_words)} unique words, {len(common_words)} common words")

        # All unique words go to training set
        train_split = unique_words.copy()

        # Split common words using stratified approach by writer
        if common_words:
            writers = [ann['writer_id'] for ann in common_words]

            # First split: train vs (val + test)
            train_common, temp_split = train_test_split(
                common_words, 
                test_size=(self.val_ratio + self.test_ratio),
                stratify=writers,
                random_state=self.random_state
            )

            # Second split: val vs test
            if temp_split:
                temp_writers = [ann['writer_id'] for ann in temp_split]
                val_split, test_split = train_test_split(
                    temp_split, 
                    test_size=self.test_ratio / (self.val_ratio + self.test_ratio),
                    stratify=temp_writers,
                    random_state=self.random_state
                )
            else:
                val_split, test_split = [], []

            train_split.extend(train_common)
        else:
            val_split, test_split = [], []

        self.splits = {
            'train': train_split,
            'val': val_split,
            'test': test_split
        }

        logger.info(f"Created splits: train={len(train_split)}, val= {len(val_split)}, test= {len(test_split)}")

    def save_jsonl_splits(self, output_dir: Path) -> Dict[str, Path]:
        """
        Save train/val/test splits as JSONL files for TrOCR training
        
        Returns:
            Dict mapping split names to file paths
        """
        if not self.splits:
            raise ValueError("No splits created. Call create_stratified_splits() first")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        for split_name, annotations in self.splits.items():
            if not annotations:
                logger.warning(f"Skipping empty {split_name} split")
                continue

            jsonl_file = output_dir / f"{split_name}.jsonl"

            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for annotation in annotations:
                    # Convert to TrOCR training format
                    trocr_format = {
                        'image': annotation['image_path'],
                        'text': annotation['ground_truth_text']
                    }
                    f.write(json.dumps(trocr_format, ensure_ascii=False) + '\n')

            saved_files[split_name] = jsonl_file
            logger.info(f"Saved {len(annotations)} samples to {jsonl_file}")

        return saved_files

def create_dataset_splits(annotations_path: Path, output_dir: Path, 
                          train_ratio: float = 0.7, val_ratio: float = 0.15,
                          test_ratio: float = 0.15, random_state: int = 42) -> Dict[str, Path]:
    """
    Main function to create stratified dataset splits from annotations
    
    Args:
        annotations_path: Path to annotations.json file
        output_dir: Directory to save train.jsonl, val.jsonl, test.jsonl
        train_ratio: Proportion for training set (default 0.7)
        val_ratio: Proportion for validation set (default 0.15)
        test_ratio: Proportion for test set (default 0.15)
        random_state: Random seed for reproducible splits
        
    Returns:
        Dictionary mapping split names to JSONL file paths
    """
    logger.info(f"Creating dataset splits from {annotations_path}")

    # Create splitter instance
    splitter = DataSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )

    # Load annotations and create splits
    splitter.load_annotations(annotations_path)
    splitter.create_stratified_splits()

    # Save as JSONL files
    saved_files = splitter.save_jsonl_splits(output_dir)

    logger.info(f"Dataset splitting complete. Files saved to {output_dir}")
    return saved_files