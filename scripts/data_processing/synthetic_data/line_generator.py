"""
Line Generator - Creates synthetic text lines from existing word images
Follows the plan: Crops words, combines 3-10 words per line from same writer
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import random
from collections import defaultdict

from config.paths import DatasetPaths
from scripts.data_processing.orchestrator.augmentation_manager import (
    AugmentationManager, 
    AugmentationConfig
)

logger = logging.getLogger(__name__)

class LineGenerator:
    """
    Generates synthetic text lines from existing word images.
    
    HARDCODED: v1 (words) → v2 (lines)
    This is a one-time conversion script.
    """
    
    def __init__(
        self,
        min_chars_per_line: int = 30,
        max_chars_per_line: int = 50,
        spacing_range: Tuple[int, int] = (8, 15),
        target_width: int = 1300,
        target_height: int = 256,
        augmentations_per_line: int = 4,
        random_seed: int = 42
    ):
        """
        Args:
            min_chars_per_line: Minimum characters per line (30, avoid over-scaling short lines)
            max_chars_per_line: Maximum characters per line (50, realistic for handwriting)
            spacing_range: Min/max pixels between words at ORIGINAL size (8-15px)
                          Will be scaled proportionally when scaled to target_width
            target_width: Target output width (1300px, matches YOLO production)
            target_height: Final image height with padding (256px)
            augmentations_per_line: Number of augmented versions per line (4)
            random_seed: Seed for reproducibility
            
        Character range (30-50) ensures scaling factor stays within 0.8x-1.3x
        This prevents over-scaling short lines or under-scaling long lines
        """
        # HARDCODED: v1 → v2
        self.source_version = 'v1'
        self.output_version = 'v2'
        
        self.min_chars_per_line = min_chars_per_line
        self.max_chars_per_line = max_chars_per_line
        self.spacing_range = spacing_range
        self.target_width = target_width
        self.target_height = target_height
        self.augmentations_per_line = augmentations_per_line
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Setup paths - HARDCODED v1 and v2
        self.source_dir = DatasetPaths.TROCR_READY_DATA / 'v1'
        self.output_dir = DatasetPaths.TROCR_READY_DATA / 'v2'
        
        if not self.source_dir.exists():
            raise ValueError(f"Source directory {self.source_dir} does not exist!")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'images_augmented').mkdir(exist_ok=True)
        
        # Augmentation manager
        self.aug_manager = AugmentationManager(AugmentationConfig(random_seed=random_seed))
        
        logger.info("LineGenerator initialized:")
        logger.info("  Source: v1 (word images)")
        logger.info("  Output: v2 (synthetic lines)")
        logger.info(f"  Char range per line: {min_chars_per_line}-{max_chars_per_line}")
        logger.info(f"  Target dimensions: {target_width}x{target_height}px")
        logger.info(f"  Spacing (original size): {spacing_range}px")
        logger.info(f"  Augmentations: {augmentations_per_line}x")
    
    def load_word_images(self) -> Dict[str, List[Tuple[Path, str]]]:
        """
        Load word images grouped by writer (from train, val, AND test sets)
        
        IMPORTANT: Only loads ORIGINAL images (not augmented versions)
        We'll augment the generated lines instead, avoiding double-augmentation.
        
        Returns:
            Dict[writer_id] -> List[(image_path, text)]
        """
        words_by_writer = defaultdict(list)
        
        # Load from ALL ground truth files to not lose data
        gt_files = ['gt_train.txt', 'gt_val.txt', 'gt_test.txt']
        total_loaded = 0
        skipped_augmented = 0
        
        for gt_filename in gt_files:
            gt_file = self.source_dir / gt_filename
            
            if not gt_file.exists():
                logger.warning(f"Ground truth file not found: {gt_file}")
                continue
            
            with open(gt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) != 2:
                        logger.warning(f"Invalid line in {gt_filename}: {line}")
                        continue
                    
                    img_rel_path, text = parts
                    
                    # SKIP augmented images - we only want originals!
                    if 'images_augmented' in img_rel_path or '_aug_' in img_rel_path:
                        skipped_augmented += 1
                        continue
                    
                    img_path = self.source_dir / img_rel_path
                    
                    if not img_path.exists():
                        logger.warning(f"Image not found: {img_path}")
                        continue
                    
                    # Extract writer ID from filename (writer01, writer05, etc.)
                    filename = img_path.name
                    if filename.startswith('writer'):
                        writer_id = filename.split('_')[0]  # 'writer01'
                    elif filename.startswith('synthetic'):
                        writer_id = 'synthetic'
                    else:
                        logger.warning(f"Could not extract writer ID from: {filename}")
                        continue
                    
                    words_by_writer[writer_id].append((img_path, text))
                    total_loaded += 1
        
        logger.info(f"Loaded {total_loaded} ORIGINAL words (skipped {skipped_augmented} augmented)")
        logger.info(f"Writers found: {len(words_by_writer)}")
        for writer_id, words in words_by_writer.items():
            logger.info(f"  {writer_id}: {len(words)} words")
        
        return words_by_writer
    
    def crop_word(self, image_path: Path) -> np.ndarray:
        """
        Crop background from word image using adaptive thresholding
        
        Uses Otsu's method to automatically find optimal threshold for each image.
        This handles varying background colors (white, light gray, etc.) and
        ensures consistent cropping regardless of scan quality.
        
        Adds small padding to avoid cutting off text edges.
        
        Returns:
            Cropped word image (numpy array) - size varies per word
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use Otsu's method for automatic threshold detection
        # This adapts to each image's characteristics (handles gray backgrounds!)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find ALL coordinates of non-zero pixels (text pixels)
        coords = cv2.findNonZero(binary)
        if coords is None:
            # Empty image or only background - return original
            return img
        
        # Compute minimal bounding box containing ALL text pixels
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add small padding (5px) to avoid cutting off text edges
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        
        # Crop to text bounding box with padding
        cropped = img[y:y+h, x:x+w]
        
        return cropped
    
    def combine_words_to_line(
        self, 
        word_images: List[np.ndarray], 
        word_texts: List[str]
    ) -> Tuple[np.ndarray, str]:
        """
        Combine word images into a text line with target width and padded height
        
        NEW Strategy (character-based):
        1. Combine words at original sizes with spacing
        2. Scale to fit target_width (1300px) proportionally
        3. Add vertical padding to reach target_height (256px)
        4. Text fills ~50-70% of height (natural, like YOLO crops)
        
        This ensures:
        - All lines have similar width (~1300px)
        - Minimal scaling variation (0.9x-1.2x due to MAX_CHARS constraint)
        - Natural text size (not over-scaled)
        - Matches production YOLO dimensions
        
        Args:
            word_images: List of cropped word images (varying sizes)
            word_texts: Corresponding word texts
        
        Returns:
            (line_image, line_text) - Combined line at target_width × target_height
        """
        # Step 1: Combine words at ORIGINAL sizes
        base_spacing = random.randint(*self.spacing_range)
        
        # Find max original height
        max_original_height = max(img.shape[0] for img in word_images)
        
        # Calculate total width at original size
        original_total_width = sum(img.shape[1] for img in word_images) + \
                              base_spacing * (len(word_images) - 1)
        
        # Create combined image at original size
        combined = np.full((max_original_height, original_total_width, 3), 255, dtype=np.uint8)
        
        # Place words bottom-aligned
        x_offset = 0
        for word_img in word_images:
            h, w = word_img.shape[:2]
            y_offset = max_original_height - h  # Bottom-align
            combined[y_offset:y_offset + h, x_offset:x_offset + w] = word_img
            x_offset += w + base_spacing
        
        # Step 2: Scale to target_width proportionally
        scale_factor = self.target_width / original_total_width
        scaled_height = int(max_original_height * scale_factor)
        scaled_width = self.target_width
        
        scaled_line = cv2.resize(combined, (scaled_width, scaled_height), 
                                interpolation=cv2.INTER_CUBIC)
        
        # Step 3: Add vertical padding to reach target_height
        if scaled_height >= self.target_height:
            # If scaled text is too tall, just crop/resize to target_height
            final_line = cv2.resize(scaled_line, (self.target_width, self.target_height),
                                   interpolation=cv2.INTER_CUBIC)
        else:
            # Add padding (center vertically)
            padding_needed = self.target_height - scaled_height
            top_padding = padding_needed // 2
            bottom_padding = padding_needed - top_padding
            
            final_line = np.full((self.target_height, self.target_width, 3), 255, dtype=np.uint8)
            final_line[top_padding:top_padding + scaled_height, :] = scaled_line
        
        # Combine texts with spaces
        line_text = ' '.join(word_texts)
        
        return final_line, line_text
    
    def generate_lines(
        self, 
        words_by_writer: Dict[str, List[Tuple[Path, str]]]
    ) -> List[Tuple[np.ndarray, str, str]]:
        """
        Generate text lines from word images (CHARACTER-BASED)
        
        NEW Strategy:
        - Adds words until total chars reach MAX_CHARS_PER_LINE (~45)
        - Groups by case type (uppercase/lowercase/mixed) for natural look
        - Results in consistent text length → minimal scaling variation
        
        Args:
            words_by_writer: Dict[writer_id] -> List[(image_path, text)]
        
        Returns:
            List[(line_image, line_text, writer_id)]
        """
        lines = []
        
        # For each writer, generate lines
        for writer_id, words in words_by_writer.items():
            if len(words) < 3:  # Need at least 3 words
                logger.warning(f"Writer {writer_id} has only {len(words)} words, skipping")
                continue
            
            # Group words by case type for more natural combinations
            uppercase_words = [(p, t) for p, t in words if t.isupper()]
            lowercase_words = [(p, t) for p, t in words if t.islower()]
            mixed_words = [(p, t) for p, t in words if not t.isupper() and not t.islower()]
            
            logger.debug(f"{writer_id}: {len(uppercase_words)} uppercase, "
                        f"{len(lowercase_words)} lowercase, {len(mixed_words)} mixed")
            
            # Generate lines for each case group
            for case_group, case_name in [
                (uppercase_words, 'uppercase'),
                (lowercase_words, 'lowercase'),
                (mixed_words, 'mixed')
            ]:
                if len(case_group) < 3:  # Need at least 3 words
                    continue
                
                # Shuffle for variation
                words_shuffled = case_group.copy()
                random.shuffle(words_shuffled)
                
                # Generate lines based on CHARACTER COUNT
                word_idx = 0
                while word_idx < len(words_shuffled):
                    line_words = []
                    total_chars = 0
                    
                    # Add words until we reach max_chars_per_line
                    while word_idx < len(words_shuffled):
                        img_path, text = words_shuffled[word_idx]
                        
                        # Check if adding this word would exceed limit
                        new_total = total_chars + len(text)
                        if new_total <= self.max_chars_per_line:
                            line_words.append((img_path, text))
                            total_chars = new_total + 1  # +1 for space
                            word_idx += 1
                        else:
                            # If line has at least 3 words, stop here
                            if len(line_words) >= 3:
                                break
                            # Otherwise, add this word anyway (avoid too-short lines)
                            line_words.append((img_path, text))
                            total_chars = new_total + 1
                            word_idx += 1
                            break
                    
                    # Need at least 3 words for a valid line
                    if len(line_words) < 3:
                        break
                    
                    # CRITICAL: Discard lines that are too short (would over-scale)
                    line_text_preview = ' '.join(text for _, text in line_words)
                    if len(line_text_preview) < self.min_chars_per_line:
                        logger.debug(f"Skipping short line ({len(line_text_preview)} chars): '{line_text_preview}'")
                        break  # Stop processing this case group (leftover words)
                    
                    try:
                        # Crop all words
                        cropped_words = [self.crop_word(img_path) for img_path, _ in line_words]
                        word_texts = [text for _, text in line_words]
                        
                        # Combine into line (will scale to target_width × target_height)
                        line_image, line_text = self.combine_words_to_line(cropped_words, word_texts)
                        
                        lines.append((line_image, line_text, writer_id))
                        
                    except Exception as e:
                        logger.warning(f"Failed to create line for {writer_id}: {e}")
                        continue
        
        logger.info(f"Generated {len(lines)} base lines")
        return lines
    
    def save_lines(
        self, 
        lines: List[Tuple[np.ndarray, str, str]],
        apply_augmentation: bool = True
    ) -> Dict[str, int]:
        """
        Save lines to disk with augmentation
        
        Args:
            lines: List[(line_image, line_text, writer_id)]
            apply_augmentation: If True, create augmented versions
        
        Returns:
            Stats dict
        """
        # Create output directories
        images_dir = self.output_dir / 'images'
        images_aug_dir = self.output_dir / 'images_augmented'
        images_dir.mkdir(parents=True, exist_ok=True)
        images_aug_dir.mkdir(parents=True, exist_ok=True)
        
        all_entries = []  # [(rel_path, text)]
        stats = {'base_lines': 0, 'augmented_lines': 0}
        
        for idx, (line_image, line_text, writer_id) in enumerate(lines, start=1):
            # Save original
            base_filename = f"{writer_id}_line_{idx:04d}.jpg"
            base_path = images_dir / base_filename
            cv2.imwrite(str(base_path), line_image)
            
            rel_path = f"images/{base_filename}"
            all_entries.append((rel_path, line_text))
            stats['base_lines'] += 1
            
            # Create augmented versions
            if apply_augmentation:
                for aug_idx in range(self.augmentations_per_line):
                    augmented = self.aug_manager.augment_image(line_image)
                    
                    aug_filename = f"{writer_id}_line_{idx:04d}_aug_{aug_idx:02d}.jpg"
                    aug_path = images_aug_dir / aug_filename
                    cv2.imwrite(str(aug_path), augmented)
                    
                    rel_path_aug = f"images_augmented/{aug_filename}"
                    all_entries.append((rel_path_aug, line_text))
                    stats['augmented_lines'] += 1
        
        # Shuffle all entries for better training
        random.shuffle(all_entries)
        
        # Split into train/val/test (70/15/15)
        total = len(all_entries)
        train_end = int(total * 0.70)
        val_end = int(total * 0.85)
        
        train_entries = all_entries[:train_end]
        val_entries = all_entries[train_end:val_end]
        test_entries = all_entries[val_end:]
        
        # Save ground truth files
        self._write_gt_file(self.output_dir / 'gt_train.txt', train_entries)
        self._write_gt_file(self.output_dir / 'gt_val.txt', val_entries)
        self._write_gt_file(self.output_dir / 'gt_test.txt', test_entries)
        
        stats['train'] = len(train_entries)
        stats['val'] = len(val_entries)
        stats['test'] = len(test_entries)
        stats['total'] = total
        
        logger.info(f"Dataset saved to: {self.output_dir}")
        logger.info(f"  Base lines: {stats['base_lines']}")
        logger.info(f"  Augmented lines: {stats['augmented_lines']}")
        logger.info(f"  Total: {stats['total']}")
        logger.info(f"  Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
        
        return stats
    
    def _write_gt_file(self, filepath: Path, entries: List[Tuple[str, str]]):
        """Write ground truth file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for rel_path, text in entries:
                f.write(f"{rel_path}\t{text}\n")
    
    def generate_dataset(self) -> Dict[str, int]:
        """
        Complete pipeline: load words → generate lines → save with augmentation
        
        Returns:
            Stats dict
        """
        logger.info("="*60)
        logger.info("Starting Line Generation Pipeline")
        logger.info("="*60)
        
        # 1. Load word images grouped by writer (ALL splits: train+val+test)
        words_by_writer = self.load_word_images()
        
        # 2. Generate text lines
        lines = self.generate_lines(words_by_writer)
        
        if not lines:
            raise ValueError("No lines generated! Check source data.")
        
        # 3. Save with augmentation
        stats = self.save_lines(lines, apply_augmentation=True)
        
        logger.info("="*60)
        logger.info("Line Generation Complete!")
        logger.info("="*60)
        
        return stats
    
    def test_single_line(self, writer_id: Optional[str] = None, target_chars: int = 40) -> Tuple[np.ndarray, str]:
        """
        Test generating ONE text line (for debugging)
        
        NEW: Uses character-based approach with min/max constraints + case grouping
        
        Args:
            writer_id: Specific writer (None = random)
            target_chars: Target character count (~40, within 30-50 range)
        
        Returns:
            (line_image, line_text)
        """
        logger.info("Testing single line generation...")
        
        # Load words
        words_by_writer = self.load_word_images()
        
        # Select writer
        if writer_id and writer_id in words_by_writer:
            selected_writer = writer_id
        else:
            selected_writer = random.choice(list(words_by_writer.keys()))
        
        words = words_by_writer[selected_writer]
        
        # GROUP BY CASE TYPE (just like generate_lines)
        uppercase_words = [(p, t) for p, t in words if t.isupper()]
        lowercase_words = [(p, t) for p, t in words if t.islower()]
        mixed_words = [(p, t) for p, t in words if not t.isupper() and not t.islower()]
        
        # Pick a case group that has enough words
        case_groups = [
            (uppercase_words, 'UPPERCASE'),
            (lowercase_words, 'lowercase'),
            (mixed_words, 'Mixed')
        ]
        # Filter to groups with at least 5 words
        valid_groups = [(words, name) for words, name in case_groups if len(words) >= 5]
        
        if not valid_groups:
            raise ValueError(f"Writer {selected_writer} doesn't have enough words in any case group")
        
        # Pick random case group
        case_group, case_name = random.choice(valid_groups)
        
        # Shuffle and pick words until we reach target chars (within min/max range)
        random.shuffle(case_group)
        selected_words = []
        total_chars = 0
        
        for img_path, text in case_group:
            if total_chars + len(text) <= self.max_chars_per_line:
                selected_words.append((img_path, text))
                total_chars += len(text) + 1  # +1 for space
                
                # Stop if we've reached target and have at least 3 words
                if total_chars >= target_chars and len(selected_words) >= 3:
                    break
            else:
                # If we're at least at min_chars and have 3+ words, stop
                if total_chars >= self.min_chars_per_line and len(selected_words) >= 3:
                    break
        
        if len(selected_words) < 3:
            raise ValueError(f"Writer {selected_writer} doesn't have enough words in {case_name} group")
        
        # Verify we're within range
        line_text_preview = ' '.join(text for _, text in selected_words)
        if len(line_text_preview) < self.min_chars_per_line:
            logger.warning(f"Test line has only {len(line_text_preview)} chars (min: {self.min_chars_per_line})")
        
        # Crop
        cropped_words = [self.crop_word(img_path) for img_path, _ in selected_words]
        word_texts = [text for _, text in selected_words]
        
        # Combine
        line_image, line_text = self.combine_words_to_line(cropped_words, word_texts)
        
        # Calculate approximate scale factor for info
        total_original_width = sum(w.shape[1] for w in cropped_words) + \
                              10 * (len(cropped_words) - 1)  # approx spacing
        scale_factor = self.target_width / total_original_width if total_original_width > 0 else 1.0
        
        logger.info(f"Generated test line from {selected_writer}:")
        logger.info(f"  Case type: {case_name}")
        logger.info(f"  Text: '{line_text}'")
        logger.info(f"  Chars: {len(line_text)}")
        logger.info(f"  Words: {len(selected_words)}")
        logger.info(f"  Image size: {line_image.shape[1]}x{line_image.shape[0]}px")
        logger.info(f"  Approx scale factor: {scale_factor:.2f}x")
        
        return line_image, line_text


def main():
    """CLI interface for line generator (HARDCODED: v1 → v2)"""
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Generate synthetic text lines from v1 word images → v2'
    )
    parser.add_argument('--test-single', action='store_true',
                       help='Test by generating a single line')
    parser.add_argument('--test-writer', type=str, default=None,
                       help='Specific writer for test (e.g., writer01)')
    parser.add_argument('--test-chars', type=int, default=40,
                       help='Target character count for test line (default: 40, range: 30-50)')
    parser.add_argument('--show-image', action='store_true',
                       help='Display test image (requires display)')
    
    args = parser.parse_args()
    
    # Initialize generator (HARDCODED v1 → v2)
    generator = LineGenerator()
    
    if args.test_single:
        # Test mode: generate one line
        line_image, line_text = generator.test_single_line(
            writer_id=args.test_writer,
            target_chars=args.test_chars
        )
        
        # Save test image to v2/test/ directory (not root!)
        test_dir = generator.output_dir / 'test'
        test_dir.mkdir(exist_ok=True)
        test_output = test_dir / 'test_line.jpg'
        cv2.imwrite(str(test_output), line_image)
        
        logger.info(f"Test image saved to: {test_output}")
        logger.info(f"Test image saved to: {test_output}")
        
        # Display image if requested
        if args.show_image:
            try:
                from PIL import Image
                img = Image.open(test_output)
                img.show()
            except Exception as e:
                logger.warning(f"Could not display image: {e}")
    else:
        # Full dataset generation
        stats = generator.generate_dataset()
        
        logger.info("\n" + "="*60)
        logger.info("FINAL STATISTICS")
        logger.info("="*60)
        for key, value in stats.items():
            logger.info(f"{key:20s}: {value:>6d}")


if __name__ == '__main__':
    main()
