#!/usr/bin/env python3
"""
Script to remove problematic segmented images and all their augmented versions.

Usage:
    python -m scripts.data_processing.remove_problematic_images --remove writer01_page08_143_EVIGHET
    python -m scripts.data_processing.remove_problematic_images --list-pattern writer01_page08_143_EVIGHET
    python -m scripts.data_processing.remove_problematic_images --interactive
"""

import argparse
from pathlib import Path
from typing import List, Tuple
try:
    # When run as module: python -m scripts.data_processing.remove_problematic_images
    from .utils.quality_control import find_all_versions, remove_files_batch
except ImportError:
    # When run standalone: python scripts/data_processing/remove_problematic_images.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.quality_control import find_all_versions, remove_files_batch


def list_problematic_pattern(base_filename: str, dataset_dir: Path) -> None:
    """List all files matching a pattern."""
    print(f"\nSearching for files matching: {base_filename}")
    
    total_found = 0
    
    # Search in all versions
    for version_dir in dataset_dir.glob("v*"):
        files = find_all_versions(base_filename, version_dir)
        if files:
            print(f"\n{version_dir.name}/:")
            for file in files:
                size_kb = file.stat().st_size // 1024
                relative_path = file.relative_to(version_dir)
                print(f"   - {relative_path} ({size_kb} KB)")
                total_found += 1
        else:
            print(f"\n{version_dir.name}/: No matching files")
    
    print(f"\nTotal: {total_found} files found")


def remove_files(base_filename: str, dataset_dir: Path, dry_run: bool = True) -> Tuple[List[Path], List[Path]]:
    """
    Remove all versions of a problematic image.
    Wrapper around remove_files_batch for backwards compatibility.
    
    Returns:
        Tuple with (removed_files, failed_files)
    """
    # Convert single filename to list
    filenames = [base_filename]
    
    # Use the shared batch function
    removed_files, failed_filenames, messages = remove_files_batch(
        filenames, dataset_dir, dry_run=dry_run, verbose=True
    )
    
    # Convert failed_filenames back to failed_files (empty since we handle at file level)
    failed_files = []  # remove_files_batch handles file-level failures differently
    
    return removed_files, failed_files


def interactive_mode(dataset_dir: Path) -> None:
    """Interactive mode for handling problematic images."""
    print("\nInteractive mode for problematic images")
    print("Enter base filename (without .jpg and without _aug_XX)")
    print("Example: writer01_page08_143_EVIGHET")
    print("Type 'quit' to exit")
    
    while True:
        user_input = input("\nFilename to handle: ").strip()
        
        if user_input.lower() in ['quit', 'q', 'exit']:
            break
            
        if not user_input:
            continue
            
        # Remove .jpg if user typed it
        base_filename = user_input.replace('.jpg', '')
        
        # List first
        list_problematic_pattern(base_filename, dataset_dir)
        
        # Ask what user wants to do
        choice = input("\nWhat do you want to do? [l]ist, [r]emove, [d]ry-run, [s]kip: ").lower()
        
        if choice in ['r', 'remove']:
            removed, failed = remove_files(base_filename, dataset_dir, dry_run=False)
            print(f"\nRemoved {len(removed)} files")
            if failed:
                print(f"Failed to remove {len(failed)} files")
        elif choice in ['d', 'dry-run']:
            removed, failed = remove_files(base_filename, dataset_dir, dry_run=True)
            print(f"\nWould remove {len(removed)} files")
        elif choice in ['l', 'list']:
            continue  # Already listed above
        else:
            print("Skipping...")


def main():
    parser = argparse.ArgumentParser(
        description="Handle problematic segmented images and their augmented versions"
    )
    
    parser.add_argument(
        '--remove',
        type=str,
        help='Remove all versions of specified base filename'
    )
    
    parser.add_argument(
        '--list-pattern',
        type=str,
        help='List all files matching specified base filename'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('dataset/trocr_ready_data'),
        help='Path to dataset directory (default: dataset/trocr_ready_data)'
    )
    
    args = parser.parse_args()
    
    # Verify that dataset directory exists
    if not args.dataset_dir.exists():
        print(f"Dataset directory not found: {args.dataset_dir}")
        return
    
    if args.interactive:
        interactive_mode(args.dataset_dir)
    elif args.list_pattern:
        list_problematic_pattern(args.list_pattern, args.dataset_dir)
    elif args.remove:
        dry_run = args.dry_run
        removed, failed = remove_files(args.remove, args.dataset_dir, dry_run)
        
        if not dry_run:
            print(f"\nSuccessfully removed {len(removed)} files")
            if failed:
                print(f"Failed to remove {len(failed)} files")
        else:
            print(f"\nWould remove {len(removed)} files")
            print("Use without --dry-run to actually remove files")
    else:
        print("Use --interactive, --list-pattern, or --remove")
        print("Example: python -m scripts.data_processing.remove_problematic_images --list-pattern writer01_page08_143_EVIGHET")


if __name__ == "__main__":
    main()