#!/usr/bin/env python3
"""
Data detection module for Swedish handwriting OCR project.
Detects new writers in originals/ directory and validates their data.
"""

from pathlib import Path
from typing import List, Dict, Set
import json
import logging

from config.paths import DatasetPaths, ensure_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scan_originals_directory() -> List[str]:
    """
    Scan originals/ directory for all writer directories.

    Returns:
        List of writer IDs found in original directory.
    """

    originals_path = DatasetPaths.ORIGINALS

    if not originals_path.exists():
        logger.warning(f"Originals directory not found {originals_path}")
        return []
    
    writers = []
    for item in originals_path.iterdir():
        if item.is_dir() and item.name.startswith('writer'):
            # Always return clean writer names without underscores for consistency
            # This handles both writer_01 and writer01 folder formats
            clean_name = item.name.replace('_', '')
            writers.append(clean_name)

    writers.sort()
    logger.info(f"Found {len(writers)} writers in originals: {writers}")
    return writers

def get_existing_writers(version_dir: Path) -> List[str]:
    """
    Read metadata.json from version directory to get existing writers.

    Args:
        version_dir: Path to version directory (e.g., trocr_ready_data/v1/)

    Returns:
        List of writer IDs included in this version
    """
    metadata_file = version_dir / "metadata.json"

    if not metadata_file.exists():
        logger.info(f"No metadata file found in {version_dir}")
        return []
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        writers = metadata.get('writers', [])
        logger.info(f"Found {len(writers)} existing writers in {version_dir.name}: {writers}")
        return writers
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error reading metadata from  {metadata_file}: {e}")
        return []
    
def get_latest_version_dir() -> Path:
    """
    Find the latest version directory (highest version number).

    Returns:
        Path to latest version directory, or None if no version exists
    """

    trocr_ready_path = DatasetPaths.TROCR_READY_DATA

    if not trocr_ready_path.exists():
        logger.info("No trocr_ready_data directory found.")
        return None
    
    version_dirs = []
    for item in trocr_ready_path.iterdir():
        if item.is_dir() and item.name.startswith('v') and item.name[1:].isdigit():
            version_dirs.append(item)

    if not version_dirs:
        logger.info("No version directories found")
        return None
    
    # Sort by version number
    version_dirs.sort(key=lambda x: int(x.name[1:]))
    latest_version = version_dirs[-1]

    logger.info(f"Latest version found: {latest_version.name}")
    return latest_version

def detect_new_writers() -> List[str]:
    """
    Compare originals/ against latest version to find new writers.

    Returns:
        List of new writer IDs that need to be processed
    """
    all_writers = scan_originals_directory()

    if not all_writers:
        logger.warning("No writers found in originals directory")
        return []
    
    latest_version_dir = get_latest_version_dir()

    if latest_version_dir is None:
        # No existing versions, all writers are new
        logger.info("No existing versions found, all writers are new")
        return all_writers
    
    existing_writers = get_existing_writers(latest_version_dir)
    existing_writers_set = set(existing_writers)
    all_writers_set = set(all_writers)

    new_writers = list(all_writers_set - existing_writers_set)
    new_writers.sort()

    logger.info(f"New writers detected: {new_writers}")
    return new_writers

def validate_new_writer_data(writer_id: str) -> bool:
    """
    Validate that writer has valid JPG files in their directory

    Args:
        writer_id: Writer ID to validate (e.g., 'writer_03')

    Returns:
        True if writer has valid data, False otherwise
    """

    # Try to find the actual folder (could be writer01 or writer_01)
    writer_path = None
    for folder_format in [writer_id, f'writer_{writer_id[6:]}']:
        potential_path = DatasetPaths.ORIGINALS / folder_format
        if potential_path.exists():
            writer_path = potential_path
            break
    
    if writer_path is None:
        logger.error(f"Writer directory not found for: {writer_id}")
        return False
    
    if not writer_path.is_dir():
        logger.error(f"Writer path is not a directory: {writer_path}")
        return False
    
    # Find JPG files
    jpg_files = list(writer_path.glob('*.jpg')) + list(writer_path.glob('*.JPG'))

    if not jpg_files:
        logger.error(f"No JPG files found for {writer_id}")
        return False
    
    valid_files = 0
    for jpg_file in jpg_files:
        try:
            file_size = jpg_file.stat().st_size
            if file_size > 1000:
                valid_files += 1
            else:
                logger.warning(f"Small file detected: {jpg_file} ({file_size})")
        except Exception as e:
            logger.error(f"Error checking file {jpg_file}: {e}")

    if valid_files == 0:
        logger.error(f"No valid JPG files found for {writer_id}")
        return False
    
    logger.info(f"Writer {writer_id} validation passed: {valid_files} valid JPG files")
    return True

def validate_all_new_writers(new_writers: List[str]) -> List[str]:
    """
    Validate all new writers and return only valid ones.

    Args: 
        new_writers: List of new writer IDs to validate

    Returns: 
        List of validated writer IDs
    """
    if not new_writers:
        logger.info("No new writers to validate")
        return []

    valid_writers = []

    for writer_id in new_writers:
        if validate_new_writer_data(writer_id):
            valid_writers.append(writer_id)
        else:
            logger.warning(f"Skipping invalid writer: {writer_id}")
            return valid_writers

    logger.info(f"Validated writers: {valid_writers}")
    return valid_writers

def get_detection_summary() -> Dict:
    """
    Get comprehensive summary of data detection results.

    Returns: 
        Dictionary with detection summary
    """

    all_writers = scan_originals_directory()
    new_writers = detect_new_writers()

    print(f"DEBUG: new_writers = {new_writers}")  # <- Debug
    valid_new_writers = validate_all_new_writers(new_writers) or []
    print(f"DEBUG: valid_new_writers = {valid_new_writers}")  # <- Debug
    print(f"DEBUG: type(valid_new_writers) = {type(valid_new_writers)}")  # <- Debug
    latest_version_dir = get_latest_version_dir()
    existing_writers = get_existing_writers(latest_version_dir) if latest_version_dir else []

    summary = {
        'all_writers': all_writers,
        'existing_writers': existing_writers,
        'new_writers': new_writers,
        'valid_new_writers': valid_new_writers,
        'latest_version': latest_version_dir.name if latest_version_dir else None,
        'total_writers': len(all_writers),
        'new_writer_count': len(valid_new_writers),
        'needs_processing': len(valid_new_writers) > 0
    }

    return summary

if __name__ == '__main__':
    # Test the module
    print("=== Data Detection Test ===")
    summary = get_detection_summary()
    
    print(f"Total writers in originals: {summary['total_writers']}")
    print(f"Existing writers: {summary['existing_writers']}")
    print(f"New writers detected: {summary['new_writers']}")
    print(f"Valid new writers: {summary['valid_new_writers']}")
    print(f"Latest version: {summary['latest_version']}")
    print(f"Processing needed: {summary['needs_processing']}")