""" 
Quality control utilities for removing problematic images.
Shared functions used by both orchestrator and standalone remove script.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def find_all_versions(base_filename: str, version_dir: Path) -> List[Path]:
    """
    Find all versions of an image (original + augmented).
    
    Args:
        base_filename: Base filename without suffix (ex: writer01_page08_143_EVIGHET)
        version_dir: Version directory (ex: v1, v2)
        
    Returns:
        List with all matching files
    """
    found_files = []
    
    # Search in images/ directory for originals
    images_dir = version_dir / "images"
    if images_dir.exists():
        original_file = images_dir / f"{base_filename}.jpg"
        if original_file.exists():
            found_files.append(original_file)
    
    # Search in images_augmented/ directory for augmented versions
    augmented_dir = version_dir / "images_augmented"
    if augmented_dir.exists():
        augmented_pattern = f"{base_filename}_aug_*.jpg"
        augmented_files = list(augmented_dir.glob(augmented_pattern))
        found_files.extend(augmented_files)
    
    return sorted(found_files)

def remove_files_batch(writer_word_pairs: List[Tuple[str, str]], dataset_dir: Path, dry_run: bool = False, verbose: bool = True, specific_version: Optional[str] = None) -> Tuple[List[Path], List[Tuple[str, str]], List[str]]:
    """
    Remove several files simultaneously based on writer:word pairs.
    
    Args:
        writer_word_pairs: List of (writer, word) tuples to remove
        dataset_dir: Dataset directory with versions
        dry_run: If True, only show which files would be removed
        verbose: If True, print messages to console (default: True for backwards compatibility)
        specific_version: If provided, only search in this version (orchestrator mode)
        
    Returns:
        Tuple with (removed_files, failed_pairs, messages)
    """
    removed_files = []
    failed_pairs = []
    messages = []
    
    # Track which files we've already processed to avoid duplicates
    processed_files = set()
    
    for writer, word in writer_word_pairs:
        try:
            files_for_pair = []
            
            # Find versions to search in
            if specific_version:
                # Search only in specific version (orchestrator mode)
                version_dirs = [dataset_dir / f"v{specific_version}"]
            else:
                # Search in all versions (standalone mode)
                version_dirs = list(dataset_dir.glob("v*"))

            # Find all files matching writer:word
            for version_dir in version_dirs:
                if version_dir.exists():
                    found_files = find_files_by_writer_word(writer, word, version_dir)
                    files_for_pair.extend(found_files)

            if not files_for_pair:
                failed_pairs.append((writer, word))
                continue

            # Remove all files for this writer:word pair (avoiding duplicates)
            pair_removed_count = 0
            for file_path in files_for_pair:
                # Skip if we've already processed this file
                if str(file_path) in processed_files:
                    continue
                
                processed_files.add(str(file_path))
                
                if dry_run:
                    messages.append(f"   Would remove: {file_path.relative_to(dataset_dir)}")
                else:
                    if file_path.exists():  # Check if file still exists
                        file_path.unlink()
                        messages.append(f"    Removed: {file_path.relative_to(dataset_dir)}")
                        pair_removed_count += 1
                
                removed_files.append(file_path)
                
            if pair_removed_count > 0:
                messages.append(f" Removed: {writer}:{word} ({pair_removed_count} files)")

        except Exception as e:
            logger.error(f"Error processing {writer}:{word}: {e}")
            failed_pairs.append((writer, word))
    
    # Print all messages at the end if verbose is enabled
    if verbose:
        for message in messages:
            print(message)
    
    return removed_files, failed_pairs, messages

def count_total_files(dataset_dir: Path) -> int:
    """
    Count total amount of images in dataset.
    
    Args:
        dataset_dir: Dataset directory with versions
        
    Returns:
        Total amount of .jpg files
    """
    total_count = 0
    
    for version_dir in dataset_dir.glob("v*"):
        # Count in images/
        images_dir = version_dir / "images"
        if images_dir.exists():
            total_count += len(list(images_dir.glob("*.jpg")))
        
        # Count in images_augmented/
        augmented_dir = version_dir / "images_augmented"
        if augmented_dir.exists():
            total_count += len(list(augmented_dir.glob("*.jpg")))
    
    return total_count


def parse_writer_word_input(user_input: str) -> List[Tuple[str, str]]:
    """
    Parse user input for writer:word format.
    
    Args:
        user_input: Comma separated string with writer:word format
        
    Returns:
        List of (writer, word) tuples
    """
    if not user_input or user_input.lower() in ['none', 'n', '']:
        return []
    
    # Split on comma and parse writer:word format
    writer_word_pairs = []
    for item in user_input.split(','):
        item = item.strip()
        if ':' in item:
            writer, word = item.split(':', 1)
            writer_word_pairs.append((writer.strip(), word.strip()))
    
    return writer_word_pairs


def find_files_by_writer_word(writer: str, word: str, version_dir: Path) -> List[Path]:
    """
    Find all files matching writer and containing word.
    
    Args:
        writer: Writer ID (e.g., "writer01")
        word: Word to search for (e.g., "STRÖM")
        version_dir: Version directory to search in
        
    Returns:
        List of matching files
    """
    found_files = []
    
    # Search in images/ directory
    images_dir = version_dir / "images"
    if images_dir.exists():
        pattern = f"{writer}_*{word}*"
        matching_files = list(images_dir.glob(pattern))
        found_files.extend(matching_files)
    
    # Search in images_augmented/ directory (if exists)
    augmented_dir = version_dir / "images_augmented"
    if augmented_dir.exists():
        pattern = f"{writer}_*{word}*"
        matching_files = list(augmented_dir.glob(pattern))
        found_files.extend(matching_files)
    
    return sorted(found_files)