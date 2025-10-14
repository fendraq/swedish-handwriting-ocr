#!/usr/bin/env python3
"""
Script för att ta bort problematiska segmenterade bilder och alla deras augmented versioner.

Usage:
    python -m scripts.data_processing.remove_problematic_images --remove writer01_page08_143_EVIGHET
    python -m scripts.data_processing.remove_problematic_images --list-pattern writer01_page08_143_EVIGHET
    python -m scripts.data_processing.remove_problematic_images --interactive
"""

import argparse
from pathlib import Path
from typing import List, Tuple


def find_all_versions(base_filename: str, search_dir: Path) -> List[Path]:
    """
    Hitta alla versioner av en bild (original + augmented).
    
    Args:
        base_filename: Basfilnamn utan filändelse (ex: writer01_page08_143_EVIGHET)
        search_dir: Directory att söka i
        
    Returns:
        Lista med alla matchande filer
    """
    # Sök efter original + augmented versioner
    patterns = [
        f"{base_filename}.jpg",           # Original
        f"{base_filename}_aug_*.jpg",     # Augmented versioner
    ]
    
    found_files = []
    for pattern in patterns:
        matches = list(search_dir.glob(pattern))
        found_files.extend(matches)
    
    return sorted(found_files)


def list_problematic_pattern(base_filename: str, dataset_dir: Path) -> None:
    """Lista alla filer som matchar ett pattern."""
    print(f"\nSöker efter filer som matchar: {base_filename}")
    
    # Sök i alla versioner
    for version_dir in dataset_dir.glob("v*"):
        images_dir = version_dir / "images"
        if not images_dir.exists():
            continue
            
        files = find_all_versions(base_filename, images_dir)
        if files:
            print(f"\n{version_dir.name}/images/:")
            for file in files:
                size_kb = file.stat().st_size // 1024
                print(f"   - {file.name} ({size_kb} KB)")
        else:
            print(f"\n{version_dir.name}/images/: Inga matchande filer")


def remove_files(base_filename: str, dataset_dir: Path, dry_run: bool = True) -> Tuple[List[Path], List[Path]]:
    """
    Ta bort alla versioner av en problematisk bild.
    
    Returns:
        Tuple med (removed_files, failed_files)
    """
    removed_files = []
    failed_files = []
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Tar bort filer för: {base_filename}")
    
    # Sök i alla versioner
    for version_dir in dataset_dir.glob("v*"):
        images_dir = version_dir / "images"
        if not images_dir.exists():
            continue
            
        files = find_all_versions(base_filename, images_dir)
        if not files:
            continue
            
        print(f"\n{version_dir.name}/images/:")
        for file in files:
            try:
                if not dry_run:
                    file.unlink()
                    print(f"   Removed: {file.name}")
                else:
                    print(f"   Would remove: {file.name}")
                removed_files.append(file)
            except Exception as e:
                print(f"   Failed to remove {file.name}: {e}")
                failed_files.append(file)
    
    return removed_files, failed_files


def interactive_mode(dataset_dir: Path) -> None:
    """Interaktiv mode för att hantera problematiska bilder."""
    print("\nInteraktiv mode för problematiska bilder")
    print("Ange basfilnamn (utan .jpg och utan _aug_XX)")
    print("Exempel: writer01_page08_143_EVIGHET")
    print("Skriv 'quit' för att avsluta")
    
    while True:
        user_input = input("\nFilnamn att hantera: ").strip()
        
        if user_input.lower() in ['quit', 'q', 'exit']:
            break
            
        if not user_input:
            continue
            
        # Ta bort .jpg om användaren skrev det
        base_filename = user_input.replace('.jpg', '')
        
        # Lista först
        list_problematic_pattern(base_filename, dataset_dir)
        
        # Fråga vad användaren vill göra
        choice = input("\nVad vill du göra? [l]ista, [r]emove, [d]ry-run, [s]kip: ").lower()
        
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
        description="Hantera problematiska segmenterade bilder och deras augmented versioner"
    )
    
    parser.add_argument(
        '--remove',
        type=str,
        help='Ta bort alla versioner av specificerat basfilnamn'
    )
    
    parser.add_argument(
        '--list-pattern',
        type=str,
        help='Lista alla filer som matchar specificerat basfilnamn'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Starta interaktiv mode'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Visa vad som skulle tas bort utan att faktiskt ta bort'
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=Path('dataset/trocr_ready_data'),
        help='Path till dataset directory (default: dataset/trocr_ready_data)'
    )
    
    args = parser.parse_args()
    
    # Verifiera att dataset directory finns
    if not args.dataset_dir.exists():
        print(f"Dataset directory hittades inte: {args.dataset_dir}")
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
        print("Använd --interactive, --list-pattern, eller --remove")
        print("Exempel: python -m scripts.data_processing.remove_problematic_images --list-pattern writer01_page08_143_EVIGHET")


if __name__ == "__main__":
    main()