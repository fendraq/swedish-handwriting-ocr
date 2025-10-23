import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass
import random
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """ Configuration for data augmentation pipeline """
    # Rotation settings
    rotation_range: Tuple[float, float] = (-5.0, 5.0)  # degrees
    rotation_probability: float = 0.7

    # Gaussian blur settings
    blur_sigma_range: Tuple[float, float] = (0.3, 0.8)
    blur_probability: float = 0.5

    # Brightness/contrast settings
    brightness_range: Tuple[float, float] = (0.85, 1.15)
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    brightness_probability: float = 0.6
    contrast_probability: float = 0.6

    # General settings
    random_seed: int = 42
    apply_during_training_only: bool = True
    version_specific: bool = True  # Different augmentation per version

    def to_dict(self) -> Dict[str, Any]: 
        """Convert to dictionary for JSON serialization"""
        return {
            'rotation_range': self.rotation_range,
            'rotation_probability': self.rotation_probability,
            'blur_sigma_range': self.blur_sigma_range,
            'blur_probability': self.blur_probability,
            'brightness_range': self.brightness_range,
            'contrast_range': self.contrast_range,
            'brightness_probability': self.brightness_probability,
            'contrast_probability': self.contrast_probability,
            'random_seed': self.random_seed,
            'apply_during_training_only': self.apply_during_training_only,
            'version_specific': self.version_specific
        }

class AugmentationManager:
    """ Manages data augmentation for TrOCR training pipeline """

    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self.rng = np.random.RandomState(self.config.random_seed)
        random.seed(self.config.random_seed)

    def apply_rotation(self, image: np.ndarray) -> np.ndarray:
        """ Apply random rotation within specified range """
        if self.rng.random() > self.config.rotation_probability:
            return image
        
        angle = self.rng.uniform(*self.config.rotation_range)

        # Get image center and rotation matrix
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation with white background
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        return rotated

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """ Apply random gaussian blur """
        if self.rng.random() > self.config.blur_probability:
            return image
        
        sigma = self.rng.uniform(*self.config.blur_sigma_range)

        kernel_size = int(2 * np.ceil(2 * sigma) + 1)

        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def apply_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """ Apply random brightness and contrast adjustments """
        result = image.copy()

        # Apply brightness adjustment
        if self.rng.random() <= self.config.brightness_probability:
            brightness_factor = self.rng.uniform(*self.config.brightness_range)
            result = cv2.convertScaleAbs(result, beta=(brightness_factor - 1.0) * 128)

        if self.rng.random() <= self.config.contrast_probability:
            contrast_factor = self.rng.uniform(*self.config.contrast_range)
            result = cv2.convertScaleAbs(result, alpha=contrast_factor, beta=0)

        return result
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """ Apply all augmentations to a single image """
        # Apply augmentations in a sequence
        augmented = image.copy()

        # 1. Rotation
        augmented = self.apply_rotation(augmented)

        # 2. Blur
        augmented = self.apply_gaussian_blur(augmented)

        # 3. Brightness/contrast
        augmented = self.apply_brightness_contrast(augmented)

        return augmented
    
    def create_augmented_dataset(self, image_paths: List[Path], output_dir: Path, 
                                 augmentations_per_image: int = 3) -> List[Path]:
        """
        Create augmented versions of images for training

        Args: 
            image_paths: List of original image paths
            output_dir: Directory to save augmented images
            augmentations_per_image: Number of augmented versions per original
        
        Returns:
            List of paths to all augmented images
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        augmented_paths = []

        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not load image: {img_path}")
                continue

            # Create multiple augmented versions
            for aug_idx in range(augmentations_per_image):
                augmented_image = self.augment_image(image)

                stem = img_path.stem
                suffix = img_path.suffix
                aug_filename = f"{stem}_aug_{aug_idx:02d}{suffix}"
                aug_path = output_dir / aug_filename

                cv2.imwrite(str(aug_path), augmented_image)
                augmented_paths.append(aug_path)

        logger.info(f"Created {len(augmented_paths)} augmented images in {output_dir}")
        return augmented_paths

    def update_annotations_with_augmented(self, original_annotations: List[Dict[str, Any]], 
                                          augmented_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Create annotations for augmented images based on original annotations

        Args:
            original_annotations: List of original annotations
            augmented_paths: List of paths to augmented images

        Returns:
            Combined list of original and augmented annotations
        """
        updated_annotations = original_annotations.copy()

        # Group augmented paths by original image name
        aug_groups = defaultdict(list)
        for aug_path in augmented_paths:
            # Extract original name from augmented filename
            # Format: original_name_aug_XX.jpg -> original_name.jpg
            aug_name = aug_path.stem
            if '_aug_' in aug_name:
                original_name = aug_name.split('_aug_')[0]
                aug_groups[original_name].append(aug_path)

        # Find matching original annotations and create augmented versions
        for original_ann in original_annotations:
            original_path = Path(original_ann['image_path'])
            original_name = original_path.stem

            if original_name in aug_groups:
                for aug_path in aug_groups[original_name]:
                    aug_annotation = original_ann.copy()
                    aug_annotation['image_path'] = str(aug_path.relative_to(aug_path.parent.parent))
                    aug_annotation['category'] = 'word_augmented' # Mark as augmented
                    aug_annotation['confidence'] = original_ann.get('confidence', 1.0) * 0.95 # Slightly lower confidence

                    updated_annotations.append(aug_annotation)

        logger.info(f"Added {len(updated_annotations) - len(original_annotations)} augmented annotations")
        return updated_annotations

    def save_config(self, output_path: Path) -> None:
        """Save augmentation configuration to JSON file"""
        config_data = {
            'augmentation_config': self.config.to_dict(),
            'creation_timestamp': str(datetime.now()),
            'version': '1.0'
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved augmentation config to {output_path}")

    @classmethod
    def load_config(cls, config_path: Path) -> 'AugmentationManager': 
        """Load augmentation configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Create AugmentationConfig from loaded data
        aug_config_dict = config_data['augmentation_config']
        config = AugmentationConfig(**aug_config_dict)

        return cls(config)

def create_augmented_training_data(version_dir: Path, config: AugmentationConfig = None, 
                                augmentations_per_image: int = 3) -> Tuple[Path, Path]:
    """
    Main function to create augmented training data for a version

    Args:
        version_dir: Path to version directory (e.g., trocr_ready_dir/v1)
        config: Augmentation configuration (uses default if None)
        augmentations_per_image: Number of augmented versions per original
        
    Returns:
        Tuple of (augmented_annotations_path, augmentation_config_path)    
    """
    logger.info(f"Creating augmented training data for {version_dir}")

    # Setup paths
    images_dir = version_dir / 'images'
    annotations_path = version_dir / 'annotations.json'
    augmented_dir = version_dir / 'images_augmented'
    aug_annotations_path = version_dir / 'annotations_augmented.json'
    aug_config_path = version_dir / 'augmentation_config.json'

    # Verify inputs exist
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    # Create augmentation manager
    manager = AugmentationManager(config)

    # Save augmentation config
    manager.save_config(aug_config_path)

    # Get all image paths
    image_paths = list(images_dir.glob('*.jpg'))
    logger.info(f"Found {len(image_paths)} images to augment")

    # Create augmented images
    augmented_paths =  manager.create_augmented_dataset(
        image_paths, augmented_dir, augmentations_per_image
    )

    # Load original annotations and create augmented annotations
    with open(annotations_path, 'r', encoding='utf-8') as f:
        original_annotations = json.load(f)

    augmented_annotations = manager.update_annotations_with_augmented(
        original_annotations, augmented_paths
    )

    # Save combined annotations
    with open(aug_annotations_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_annotations, f, indent=2, ensure_ascii=False)

    logger.info(f"Augmentation complete. Total annotations: {len(augmented_annotations)}")
    logger.info(f"Original: {len(original_annotations)}, Augmented: {len(augmented_annotations) - len(original_annotations)}")

    return aug_annotations_path, aug_config_path
