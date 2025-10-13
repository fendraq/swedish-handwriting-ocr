#!/usr/bin/env python3
"""
Version management module for Swedish handwriting OCR project.
Handles dataset versioning (v1, v2, v3), data copying and version cleanup
"""

import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

from config.paths import DatasetPaths, ensure_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_latest_version_number() -> Optional[str]: 
    """
    Find the latest version number (highest version number)

    Returns:
        Latest version string (e.g., 'v3') or None if no version exists
    """

    trocr_ready_path = DatasetPaths.TROCR_READY_DATA

    if not trocr_ready_path.exists():
        logger.info("No trocr_ready_data directory found")
        return None
    
    version_dirs = []
    for item in trocr_ready_path.iterdir():
        if item.is_dir() and item.name.startswith('v') and item.name[1:].isdigit():
            version_number = int(item.name[1:])
            version_dirs.append((version_number, item.name))

    if not version_dirs:
        logger.info("No version directories found")
        return None
    
    # Sort by version number and get the highest
    version_dirs.sort(key=lambda x: x[0])
    latest_version = version_dirs[-1][1]

    logger.info(f"Latest version found: {latest_version}")
    return latest_version

def create_new_version(writers: List[str] = None, description: str = "") -> Path:
    """
    Create a new version ditrectory with metadata.

    Args:
        writers: List of writer IDs to include in this version
        description: Optional description of this version

    Returns:
        Path to new version directory
    """

    # Ensure trocr_ready_data directory exists
    ensure_dir(DatasetPaths.TROCR_READY_DATA)

    # Get next version number
    latest_version = get_latest_version_number()
    if latest_version is None:
        new_version_num = 1
    else:
        new_version_num = int(latest_version[1:]) + 1

    new_version = f"v{new_version_num}"
    new_version_path = DatasetPaths.TROCR_READY_DATA / new_version

    if writers is None:
        from .data_detector import detect_new_writers
        writers = detect_new_writers()

    logger.info(f"Creating new version {new_version}")

    # Create version directory structure
    ensure_dir(new_version_path)
    ensure_dir(new_version_path / 'images')

    # Creat metadata.json
    metadata = {
        'version': new_version,
        'created': datetime.now().isoformat(),
        'writers': sorted(writers),
        'total_writers': len(writers),
        'description': description,
        'status': "in_progress"
    }

    metadata_file = new_version_path / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Created version {new_version} with {len(writers)} writers")
    return new_version_path

def  copy_existing_data(from_version: str, to_version: str) -> bool:
    """
    Copy existing processed data from previous version to new version

    Args:
        from_version: Source version (e.g., 'v1')
        to_version: Target version (e.g., 'v2')
    """
    from_path = DatasetPaths.TROCR_READY_DATA / from_version
    to_path = DatasetPaths.TROCR_READY_DATA / to_version

    if not from_path.exists():
        logger.warning(f"Source version {from_version} does not exist")
        return False
    
    if not to_path.exists():
        logger.warning(f"Target version {to_version} does not exist")
        return False
    
    from_images = from_path / 'images'
    to_images = to_path / 'images'

    if not from_images.exists():
        logger.info(f"No images directory in {from_version}")
        return True
    
    try:
        # Copy all images from previous version
        copied_count = 0
        for image_file in from_images.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', 'png']:
                dest_file = to_images / image_file.name
                if not dest_file.exist():
                    shutil.copy2(image_file, dest_file)
                    copied_count += 1
                else:
                    logger.debug(f"Skipping existing file: {image_file.name}")

        logger.info(f"Copied {copied_count} images from {from_version} to {to_version}")
        return True
    except Exception as e:
        logger.error(f"Error copying data from {from_version} to {to_version}: {e}")
        return False

def cleanup_old_versions(keep_count: int = 3) -> List[str]:
    """
    Remove old versions, keep only the latest N versions

    Args:
        keep_count: Number of versions to keep (default: 3)

    Retuns: 
        List of removed version names
    """
    trocr_ready_path = DatasetPaths.TROCR_READY_DATA

    if not trocr_ready_path.exists():
        logger.info("No trocr_ready_data directory found")
        return []
    
    # Get all versiondirectories
    version_dirs = []
    for item in trocr_ready_path.iterdir():
        if item.is_dir() and item.name.startswith('v') and item.name[1:].isdigit():
            version_number = int(item.name[1:])
            version_dirs.append((version_number, item.name, item))

    if len(version_dirs) <= keep_count:
        logger.info(f" Only {len(version_dirs)} versions found, nothing to cleanup")
        return []
    
    version_dirs.sort(key=lambda x: x[0])

    # Determin versions to remove
    versions_to_remove = version_dirs[:-keep_count]
    removed_versions = []

    for version_num, version_name, version_path in versions_to_remove:
        try:
            logger.info(f"Removing old version: {version_name}")
            shutil.rmtree(version_path)
            removed_versions.appen(version_name)
        except Exception as e:
            logger.error(f"Error removing version {version_name}: {e}")

    logger.info(f"Cleanup complete. Removed {len(removed_versions)} old versions: {removed_versions}")
    return removed_versions

def update_current_symlink(version: str) -> bool:
    """
    Update the 'current' symlink to point to the specified version.

    Args:
        version: Version to point to (e.g., 'v2')

    Returns:
        True if sucessful, False if faild
    """

    version_path = DatasetPaths.TROCR_READY_DATA / version
    current_link = DatasetPaths.CURRENT_VERSION

    if not version_path.exists():
        logger.error(f"Version {version} does not exist")
        return False

    try: 
        # Remove existing symlink if it exists
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()

        current_link.symlink_to(version, target_is_directory=True)

        logger.info(f"Updated current symlink to point to {version}")
        return True
    except Exception as e:
        logger.error(f"Error updating current symlink {e}")
        return False

def get_version_info(version: str) -> Optional[Dict]:
    """
    Get metadata information for a specific version.

    Args:
        version: Version name (e.g., 'v2')

    Returns:
        Version metadata dictionary or None if not found.
    """
    version_path = DatasetPaths.TROCR_READY_DATA / version
    metadata_file = version_path / 'metadata.json'

    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error reading metadata for {version}: {e}")
        return None
    
if __name__ == '__main__':
    # Test module
    print("=== Version Manager Test ===")

    # Test version detection
    latest = get_latest_version_number()
    print(f"Latest version: {latest}")
    
    # Test version creation
    test_writers = ['writer01', 'writer02', 'writer03']
    new_version = create_new_version(test_writers, "Test version creation")
    print(f"Created version: {new_version}")
    
    # Test symlink update
    success = update_current_symlink(new_version)
    print(f"Symlink updated: {success}")
    
    # Test version info
    info = get_version_info(new_version)
    if info:
        print(f"Version info: {info['writers']} ({info['total_writers']} writers)")