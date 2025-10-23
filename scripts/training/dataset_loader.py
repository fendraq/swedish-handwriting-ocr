"""
SwedishHandwritingDataset - Dataset loader for TrOCR training
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor
from config.paths import DatasetPaths
from scripts.data_processing.orchestrator.version_manager import get_latest_version_number

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SwedishHandwritingDataset(Dataset):
    """
    PyTorch Dataset for Swedish handwriting OCR training

    Reads ground truth files created by the orchestrator and converts
    them to the format required for TrOCR fine-tuning.

    Args:
        gt_file_path (str): Path to ground truth file (gt_train.txt, etc.)
        processor (TrOCRProcessor, optional): Pre-loaded processor
        image_base_path (str, optional): Base path for images
        dry_run (bool): If True, only loads first 10 samples for testing
    """

    def __init__(
            self,
            gt_file_path: str,
            processor: Optional[TrOCRProcessor] = None,
            image_base_path: Optional[str] = None,
            dry_run: bool = False
    ):
        if image_base_path is None:
            latest_version = get_latest_version_number()
            if latest_version is None:
                raise FileNotFoundError("No version directory found")
            self.image_base_path = DatasetPaths.TROCR_READY_DATA / latest_version
            logger.info(f"Using latest version: {latest_version} at {self.image_base_path}")
        else:
            self.image_base_path = Path(image_base_path)

        self.gt_file_path = Path(gt_file_path)

        self.dry_run = dry_run

        # Load or create TrOCR processor
        if processor is None:
            logger.info("Loading TrOCR processor...")
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        else:
            self.processor = processor

        # Load and validate dataset
        self.data = self._load_gt_file()

        # Count original vs augmented based on filename patterns
        original_count = len([item for item in self.data if not str(item[0]).endswith('_aug')])
        augmented_count = len([item for item in self.data if str(item[0]).endswith('_aug')])

        self._validate_dataset()

        logger.info(f"Loaded {len(self.data)} samples from {self.gt_file_path}")
        logger.info(f"  - {original_count} original images")
        logger.info(f"  - {augmented_count} augmented images")
        logger.info(f"  - Augmentation ratio. {augmented_count/original_count:.1f}x")
        if self.dry_run:
            logger.info("DRY RUN MODE: Limited to first 10 samples")

    def __len__(self) -> int:
        """ Return number of samples in dataset """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample in HuggingFace format.
        Handles both original and augmented images

        Args.
            idx (int): Sample index

        Returns:
            dict: {'pixel_values': image_tensor, 'labels': text_tokens}

        Raises:
            IndexError: If idx is out of range
            IOError. If Image cannot be loaded
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")

        # Get image path and text for this sample
        img_path, text = self.data[idx]

        try:
            # 1. Load and preprocess image
            image = Image.open(img_path).convert('RGB')

            # 2. Process image through TrOCR processor 
            inputs = self.processor(
                images=image, 
                text=text,
                return_tensors='pt',
                padding=True, 
                truncation=True,
                max_length=512
            )

            return {
                'pixel_values': inputs['pixel_values'].squeeze(),
                'labels': inputs['labels'].squeeze()
            }
        
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({img_path}): {e}")
            raise IOError(f"Cannot load sample {idx}: {e}")

    def _load_gt_file(self) -> list[tuple[Path, str]]:
        """
        Load and parse ground truth file, including augmented image variants.

        For each original image, also includes corresponding augmented variants
        (aug_00.jpg, aug_01.jpg, aug_02.jpg) if they exist.

        Returns:
            list[tuple[Path, str]]: List of (image_path, text) pairs

        Raises:
            FileNotFoundError: If gt file doesn't exist
            ValueError: if file format is invalid
        """
        if not self.gt_file_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.gt_file_path}")

        data = []
        with open(self.gt_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: 
                    continue

                try: 
                    # split on tab: 'image_path\ttext'
                    parts = line.split('\t')
                    if len(parts) != 2:
                        raise ValueError(f"Expected 2 parts, got {len(parts)}")

                    img_path_str, text = parts

                    # Handle original image path
                    img_path = Path(img_path_str)
                    if not img_path.is_absolute() and self.image_base_path:
                        img_path = self.image_base_path / img_path

                    data.append((img_path, text))

                    # Dry run: Limit to 10 first samples (not counting augmented)
                    if self.dry_run and len(data) >= 10:
                        logger.info("Dry run: stopping at 10 samples")
                        break

                except ValueError as e:
                    logger.error(f"Invalid format at line {line_num}: {line}")
                    logger.error(f"Error: {e}")
                    raise ValueError(f"Invalid gt file format at line {line_num}")

        if not data:
            raise ValueError(f"No valid data found in {self.gt_file_path}")

        return data
    
    def _validate_dataset(self) -> None:
        """
        Validate that all image files exist and are readable.

        Raises: 
            FileNotFoundError: If any image file is missing
            ValueError: If any image file is corrupted
        """

        missing_files = []
        corrupt_files = []

        for img_path, text in self.data:
            # Check if file exists
            if not img_path.exists():
                missing_files.append(str(img_path))
                continue

            # Try to open image to check if it's valid
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                corrupt_files.append(f"{img_path}: {e}")

        # Report issues
        if missing_files:
            logger.error(f"Missing image files: {missing_files[:5]}...")
            raise FileNotFoundError(f"Found {len(missing_files)} missing image files")

        if corrupt_files:
            logger.error(f"Corrupt image files: {corrupt_files[:5]}...")
            raise ValueError(f"Found {len(corrupt_files)} corrupt image files")

        logger.info(f"Dataset validation successful: {len(self.data)} valid samples")

    