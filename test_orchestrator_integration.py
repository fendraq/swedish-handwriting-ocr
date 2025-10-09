#!/usr/bin/env python3
"""
Test script for orchestrator integration.
Tests the complete pipeline: data_detector → segmentation_runner → flat output with 384x384.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_version_for_testing():
    """Reset latest version to test complete pipeline"""
    logger.info("=== RESETTING VERSION FOR TESTING ===")
    
    try:
        from scripts.data_processing.orchestrator.version_manager import get_latest_version_number
        from config.paths import DatasetPaths
        
        # Get latest version
        latest_version = get_latest_version_number()
        if latest_version == 0:
            logger.info("No versions exist yet")
            return
        
        version_dir = DatasetPaths.TROCR_READY_DATA / f"v{latest_version}"
        metadata_file = version_dir / "metadata.json"
        
        if metadata_file.exists():
            import json
            
            # Load current metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Backup original processed_writers
            original_writers = metadata.get('processed_writers', [])
            logger.info(f"Backing up {len(original_writers)} processed writers")
            
            # Reset processed_writers to empty list
            metadata['processed_writers'] = []
            logger.info(f"Reset processed_writers in {metadata_file}")
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Clear images directory
            images_dir = version_dir / "images"
            if images_dir.exists():
                import shutil
                shutil.rmtree(images_dir)
                images_dir.mkdir(exist_ok=True)
                logger.info(f"Cleared images directory: {images_dir}")
            
            return original_writers
        
    except Exception as e:
        logger.error(f"Version reset failed: {e}")
        return []

def restore_version_after_testing(original_writers):
    """Restore version metadata after testing"""
    logger.info("=== RESTORING VERSION AFTER TESTING ===")
    
    try:
        from scripts.data_processing.orchestrator.version_manager import get_latest_version_number
        from config.paths import DatasetPaths
        
        latest_version = get_latest_version_number()
        if latest_version == 0:
            return
        
        version_dir = DatasetPaths.TROCR_READY_DATA / f"v{latest_version}"
        metadata_file = version_dir / "metadata.json"
        
        if metadata_file.exists() and original_writers:
            import json
            
            # Load current metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Restore original processed_writers
            metadata['processed_writers'] = original_writers
            logger.info(f"Restored {len(original_writers)} processed writers")
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
    except Exception as e:
        logger.error(f"Version restore failed: {e}")

def test_data_detection():
    """Test step 1: Data detection"""
    logger.info("=== TESTING DATA DETECTION ===")
    
    try:
        from scripts.data_processing.orchestrator.data_detector import detect_new_writers
        from config.paths import DatasetPaths
        
        new_writer_ids = detect_new_writers()
        logger.info(f"Found {len(new_writer_ids)} new writers: {new_writer_ids}")
        
        if not new_writer_ids:
            logger.warning("No new writers found - cannot proceed with segmentation test")
            return None
        
        # Convert writer IDs to dictionaries with path information
        writers = []
        for writer_id in new_writer_ids:
            writer_path = DatasetPaths.ORIGINALS / writer_id
            writers.append({
                'writer_id': writer_id,
                'path': str(writer_path)
            })
        
        return writers
        
    except Exception as e:
        logger.error(f"Data detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_version_creation():
    """Test step 2: Create test version"""
    logger.info("=== TESTING VERSION CREATION ===")
    
    try:
        from scripts.data_processing.orchestrator.version_manager import create_new_version
        
        version_dir = create_new_version(description="integration_test")
        logger.info(f"Created test version: {version_dir}")
        
        logger.info(f"Base version directory: {version_dir}")
        
        return version_dir
        
    except Exception as e:
        logger.error(f"Version creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_segmentation_single_writer(writers, output_dir):
    """Test step 3: Segmentation for single writer"""
    logger.info("=== TESTING SINGLE WRITER SEGMENTATION ===")
    
    try:
        from scripts.data_processing.orchestrator.segmentation_runner import run_segmentation_for_writer
        
        # Test with first writer
        test_writer = writers[0]
        writer_id = test_writer['writer_id']
        writer_path = Path(test_writer['path'])
        
        logger.info(f"Testing with writer: {writer_id}")
        logger.info(f"Writer path: {writer_path}")
        logger.info(f"Output directory: {output_dir}")
        
        results = run_segmentation_for_writer(
            writer_path=writer_path,
            output_dir=output_dir,
            writer_id=writer_id,
            enable_references=True,
            enable_visualization=False
        )
        
        logger.info(f"Segmentation completed!")
        logger.info(f"Results: {len(results)} images processed")
        
        total_words = sum(len(word_list) for word_list in results.values())
        logger.info(f"Total words segmented: {total_words}")
        
        return results
        
    except Exception as e:
        logger.error(f"Single writer segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def verify_output_structure(output_dir, results):
    """Test step 4: Verify output structure and format"""
    logger.info("=== VERIFYING OUTPUT STRUCTURE ===")
    
    try:
        import cv2
        
        # Check that base output directory exists
        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            return False
        
        # The actual images should be in output_dir/images/
        images_dir = output_dir / "images"
        if not images_dir.exists():
            logger.error(f"Images directory does not exist: {images_dir}")
            return False
        
        # List all files in the images directory
        all_files = list(images_dir.glob("*"))
        jpg_files = list(images_dir.glob("*.jpg"))
        txt_files = list(images_dir.glob("*.txt"))
        
        logger.info(f"Total files in images directory: {len(all_files)}")
        logger.info(f"JPG files: {len(jpg_files)}")
        logger.info(f"TXT files: {len(txt_files)}")
        
        if len(jpg_files) == 0:
            logger.error("No JPG files found in images directory!")
            return False
        
        # Check filename format
        sample_jpg = jpg_files[0]
        filename_parts = sample_jpg.stem.split('_')
        logger.info(f"Sample filename: {sample_jpg.name}")
        logger.info(f"Filename parts: {filename_parts}")
        
        # Expected format: writer_id_pageXX_word_id_text
        if len(filename_parts) < 4:
            logger.warning(f"Unexpected filename format: {sample_jpg.name}")
        
        # Check image dimensions (should be 384x384)
        sample_image = cv2.imread(str(sample_jpg))
        if sample_image is not None:
            height, width = sample_image.shape[:2]
            logger.info(f"Sample image dimensions: {width}x{height}")
            
            if width == 384 and height == 384:
                logger.info("✅ Images are correctly sized to 384x384!")
            else:
                logger.warning(f"❌ Images are not 384x384: {width}x{height}")
        
        # Check that txt files match jpg files
        if len(txt_files) == len(jpg_files):
            logger.info("✅ Equal number of JPG and TXT files")
        else:
            logger.warning(f"❌ Mismatch: {len(jpg_files)} JPG, {len(txt_files)} TXT")
        
        return True
        
    except Exception as e:
        logger.error(f"Output verification failed: {e}")
        return False

def main():
    """Run complete integration test"""
    logger.info("Starting orchestrator integration test...")
    
    # Step 0: Reset version for testing
    original_writers = reset_version_for_testing()
    
    try:
        # Step 1: Test data detection (now should find writers as "new")
        writers = test_data_detection()
        if not writers:
            logger.error("Cannot proceed without detected writers")
            return False
        
        # Step 2: Create test version
        output_dir = test_version_creation()
        if not output_dir:
            logger.error("Cannot proceed without version directory")
            return False
        
        # Step 3: Test segmentation
        results = test_segmentation_single_writer(writers, output_dir)
        if not results:
            logger.error("Segmentation test failed")
            return False
        
        # Step 4: Verify output
        verification_success = verify_output_structure(output_dir, results)
        
        if verification_success:
            logger.info("🎉 INTEGRATION TEST PASSED!")
            logger.info(f"Output saved to: {output_dir}")
            return True
        else:
            logger.error("❌ INTEGRATION TEST FAILED!")
            return False
    
    finally:
        # Always restore original state
        restore_version_after_testing(original_writers)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)