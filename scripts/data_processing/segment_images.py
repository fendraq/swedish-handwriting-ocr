#!/usr/bin/env python3
"""
Image segmentation script for Swedish handwriting dataset.
Segments scanned template images into individual word images using metadata coordinates.

Usage:
    python segment_images.py --metadata path/to/metadata.json --images path/to/images --output path/to/output --writer-id writer_001
"""

from image_segmentation import ImageSegmenter
import argparse
import sys
from pathlib import Path

def validate_inputs(args):
    """
    Validates input parameters.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        SystemExit: If validation fails.
    """
    # Check meta data file
    if not Path(args.metadata).exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)
    
    # Check image folder
    if not Path(args.images).exists():
        print(f"Error: Images directory not found: {args.images}")
        sys.exit(1)
    
    # Check for JPG in image folder
    images_path = Path(args.images)
    jpg_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.JPG"))
    if not jpg_files:
        print(f"Error: No JPG files found in {args.images}")
        sys.exit(1)
    
    print("Validation passed:")
    print(f"  Metadata: {args.metadata}")
    print(f"  Images: {args.images} ({len(jpg_files)} JPG files)")
    print(f"  Output: {args.output}")
    print(f"  Writer ID: {args.writer_id}")

def main():
    """
    Main funktion for image segmentation
    """
    # Command line arguments
    parser = argparse.ArgumentParser(
        description="Segment scanned handwriting images into individual words",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python segment_images.py --metadata ../../dataset/templates/generated_templates/complete_template_metadata.json --images /path/to/scanned/images --output ../../dataset/segmented_words --writer-id writer_001
        """
    )

    parser.add_argument(
        "--metadata", 
        required=True,
        help="Path to metadata JSON file from PDF generation"
    )
    
    parser.add_argument(
        "--images", 
        required=True,
        help="Directory containing scanned images (JPG format)"
    )

    parser.add_argument(
        "--output", 
        required=True,
        help="Output directory for segmented word images"
    )

    parser.add_argument(
        "--writer-id", 
        default="writer_001",
        help="Identifier for the person who wrote this (default: writer_001)"
    )
    
    args = parser.parse_args()

    validate_inputs(args)

    try:
        print("\n" + "="*50)
        print("STARTING IMAGE SEGMENTATION")
        print("="*50)

        segmenter = ImageSegmenter(args.metadata, args.output)

        results = segmenter.segment_multiple_pages(args.images, args.writer_id)

        total_images = len(results)
        total_words = sum(len(word_list) for word_list in results.values())
        successful_images = sum(1 for word_list in results.values() if len(word_list) > 0)
        
        print("\n" + "="*50)
        print("SEGMENTATION COMPLETE")
        print("="*50)
        print(f"Processed images: {total_images}")
        print(f"Successful images: {successful_images}")
        print(f"Total words segmented: {total_words}")
        print(f"Output directory: {args.output}")
        
        if total_words == 0:
            print("\nWarning: No words were successfully segmented!")
            print("Check your image files and metadata coordinates.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nSegmentation interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError during segmentation: {e}")
        print("Check your input files and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()

    # '../../../dataset/templates/generated_templates/complete_tempate_metadata.json'