"""
Segmentation runner module for orchestrator pipeline.
Provides programmatic interface to ImageSegmentor for automated data processing.
"""

import logging
from pathlib import Path
from typing import Dict, List

from config.paths import get_template_metadata, ensure_dir
from ..image_segmentation import ImageSegmenter

def get_metadata_path() -> Path:
    """
    Get path to template metadata file.

    Returns:
        Path to complete_template_metadata.json

    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = get_template_metadata()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Template metadata not found: {metadata_path}")
    return metadata_path

def run_segmentation_for_writer(
    writer_path: Path,
    output_dir: Path,
    writer_id: str,
    enable_references: bool = True,
    enable_visualization: bool = False
) -> Dict[str, List[str]]:
    """
    Run segmentation for a specific writer with orchestator integration

    Args:
        writer_path: Path to writer's original images directory
        output_dir: Output directory for segmented images (should be trocr_ready_data/vX/images)
        writer_id: Writer identifier (e.g., 'writer01')
        enable_references: Use reference marker detection for coordinate transformation
        enable_visualization: Generate visualization images for debugging

    Retuns:
        Dictionary mapping source images to list of segmented file paths

    Raises: 
        FileNotFoundError: If writer directory or metadata not found
        ValueError: if no valid images found
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Starting segmentation for {writer_id}")
    logger.info(f"Writer path: {writer_path}")
    logger.info(f"Output directory: {output_dir}")

    # If valid inputs
    if not writer_path.exists():
        raise FileNotFoundError(f"Writer path not found: {writer_path}")
    if not writer_path.is_dir():
        raise ValueError(f"Writer path is not a directory: {writer_path}")

    metadata_path = get_metadata_path()
    logger.info(f"Using metadata: {metadata_path}")

    ensure_dir(output_dir)

    # Initialize ImageSegmentor
    viz_output = None
    if enable_visualization:
        viz_output = str(output_dir.parent / f"{output_dir.name}_visualizations")

    segmenter = ImageSegmenter(
        metadata_path=str(metadata_path),
        output_dir=str(output_dir),
        use_references=enable_references,
        enable_visualization=enable_visualization,
        viz_output=viz_output
    )

    # Run segmentation
    try:
        results = segmenter.segment_multiple_pages(
            images_dir=str(writer_path),
            writer_id=writer_id
        )

        total_images = len(results)
        total_words = sum(len(word_list) for word_list in results.values())
        successful_images = sum(1 for word_list in results.values() if len(word_list) > 0)

        logger.info(f"Segmentation completed for {writer_id}")
        logger.info(f"Processed images: {total_images}")
        logger.info(f"Successful images: {successful_images}")
        logger.info(f"Total words segmented: {total_words}")

        if total_words == 0:
            logger.warning(f"No words were segmented for {writer_id}")

        return results
    
    except Exception as e:
        logger.error(f"Segmentation failed for {writer_id}: {e}")
        raise
    
def run_segmentation_for_multiple_writers(
    writers_data: List[Dict],
    output_dir: Path,
    enable_references: bool = True,
    enable_visualization: bool = False
) -> Dict[str, Dict[str, List[str]]]:
    """
    Run segmentation for multiple writers from data_detector output.

    Args:
        writers_data: List of writer dictionaries from data_detector.detect_new_writers()
                     Each dict should have 'writer_id' and 'path' keys
        output_dir: Base output directory (trocr_ready_data/vX/images)
        enable_references: Use reference marker detection
        enable_visualization: Generate visualization images
        
    Returns:
        Dictionary mapping writer_id to segmentation results
        Format: {writer_id: {source_image: [segmented_files]}}
    """
    logger = logging.getLogger(__name__)
    all_results = {}

    logger.info(f"Starting segmentation for {len(writers_data)} writers")

    for writer_info in writers_data:
        writer_id = writer_info['writer_id']
        writer_path = Path(writer_info['path'])

        logger.info(f"Processing writer: {writer_id}")

        try:
            results = run_segmentation_for_writer(
                writer_path=writer_path,
                output_dir=output_dir,
                writer_id=writer_id,
                enable_references=enable_references,
                enable_visualization=enable_visualization
            )
            all_results[writer_id] = results

        except Exception as e:
            logger.error(f"Failed to process {writer_id}: {e}")
            all_results[writer_id] = {}

    # Summary
    total_writers = len(writers_data)
    successful_writers = sum(1 for results in all_results.values() if results)
    total_words = sum(
        sum(len(word_list) for word_list in writer_results.values())
        for writer_results in all_results.values()
    )

    logger.info("Multi-writer segmentation completed:")
    logger.info(f"Total writers: {total_writers}")
    logger.info(f"Successful writers: {successful_writers}")
    logger.info(f"Total words segmented: {total_words}")

    return all_results

class SegmentationRunner:
    """
    Orchestrator wrapper for ImageSegmentor with integrated preprocessing.
    Provides stateful interface for batch processing and configuration.
    """

    def __init__(self, enable_references: bool = True, enable_visualization: bool = False):
        """
        Initialize segmentation runner

        Args:
            enable_references: Use reference marker detection for coordinate transformation
            enable_visualization: Generate visualization images for debugging
        """
        self.enable_references = enable_references
        self.enable_visualization = enable_visualization
        self.logger = logging.getLogger(__name__)
        self.metadata_path = None

    def get_metadata_path(self) -> Path:
        """
        Get and cache metadata path.

        Returns:
            Path to template metadata file.
        """
        if self.metadata_path is None:
            self.metadata_path = get_metadata_path()
        return self.metadata_path
    
    def process_writer(self, writer_path: Path, output_dir: Path, writer_id: str) -> Dict[str, List[str]]:
        """
        Process single writer using the runner's configuration

        Args:
            writer_path: Path to writer's images
            output_dir: Output directory for results
            writer_id: Writer identifier
            
        Returns:
            Segmentation results
        """
        return run_segmentation_for_writer(
            writer_path=writer_path,
            output_dir=output_dir,
            writer_id=writer_id,
            enable_references=self.enable_references,
            enable_visualization=self.enable_visualization
        )
    
    def process_multiple_writers(self, writers_data: List[Dict], output_dir: Path) -> Dict[str, Dict[str, List[str]]]:
        """
        Process multiple writers using the runner's configuration.
        
        Args:
            writers_data: List of writer data from data_detector
            output_dir: Output directory for results
            
        Returns:
            Combined results for all writers
        """
        return run_segmentation_for_multiple_writers(
            writers_data=writers_data,
            output_dir=output_dir,
            enable_references=self.enable_references,
            enable_visualization=self.enable_visualization
        )
