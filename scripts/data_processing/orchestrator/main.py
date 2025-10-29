import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from datetime import datetime
import shutil

from config.paths import DatasetPaths, get_template_metadata
from .data_detector import detect_new_writers
from .version_manager import create_new_version, get_latest_version_number, copy_existing_data
from .segmentation_runner import run_segmentation_for_multiple_writers
from .annotation_creator import create_annotations_for_version
from .dataset_splitter import create_dataset_splits
from .augmentation_manager import create_augmented_training_data
from ..utils import remove_files_batch, count_total_files, parse_writer_word_input
from ..synthetic_data.synthetic_data_creator import generate_synthetic_data

logger = logging.getLogger(__name__)

class PipelineConfig:
    """ Configuration for complete pipeline execution """
    def __init__(self):
        self.originals_dir = DatasetPaths.ORIGINALS
        self.trocr_ready_dir = DatasetPaths.TROCR_READY_DATA
        self.metadata_file = get_template_metadata()

        # Pipeline settings
        self.auto_detect = True
        self.apply_augmentation = True
        self.augmentations_per_image = 3
        self.keep_versions = 3 
        self.resume_on_failure = True

        # Split ratios
        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.test_ratio = 0.15

        # Logging
        self.log_level = logging.INFO
        self.save_reports = True

class PipelineRunner:
    """ Main orchestrator for completed TrOCR data pipeline """    

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.current_version = None
        self.new_writers = []
        self.pipeline_start_time = None
        self.setup_logging()

    def setup_logging(self):
        """ Configure logging for pipeline execution """
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"pipeline_{timestamp}.log"

        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logger.info(f"Pipeline logging started. Log file: {log_file}")

    def run_complete_pipeline(self, writers: Optional[List[str]]= None, 
                              dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute complete pipeline from original raw scans to TrOCR-ready data

        Args:
            writers: Specific writers to process (None for auto-detect)
            dry_run: If True, show what would happen without executing

        Returns:
            Pipeline execution report
        """
        self.pipeline_start_time = datetime.now()
        logger.info(f"Starting complete TrOCR pipeline (dry_run={dry_run})")

        try:
            # Pipeline steps
            report = {
                'start_time': self.pipeline_start_time.isoformat(),
                'steps': {},
                'status': 'running'
            }

            # 1 Detect new writers:
            self._step_detect_writers(writers, dry_run, report)

            # 2 Create new version:
            self._step_create_version(dry_run, report)

            # 3 Run segmentation for new writers
            self._step_run_segmentation(dry_run, report)
            
            # 3.4 Generate synthetic data
            self._step_generate_synthetic_data(dry_run, report)

            # 3.5 Manual quality control of images
            self._step_interactive_quality_control(dry_run, report)

            # 4 Create annotations
            self._step_create_annotations(dry_run, report)

            # 5 Apply augmentation (optional)
            if self.config.apply_augmentation:
                self._step_apply_augmentation(dry_run, report)

            # 6 Create dataset splits
            self._step_create_splits(dry_run, report)

            # 7 Validate and cleanup
            self._step_validate_and_cleanup(dry_run, report)

            report['status'] = 'completed'
            report['end_time'] = datetime.now().isoformat()

            logger.info("Pipeline completed successfully!")
            return report

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            report['status'] = 'failed'
            report['error'] = str(e)
            report['end_time'] = datetime.now().isoformat()
            raise

    def _step_detect_writers(self, writers: Optional[List[str]], dry_run: bool, 
                             report: Dict[str, Any]) -> None:
        """ Step 1: Detect new writers to process """
        logger.info("Step 1: Detecting writers to process...")
        step_start = datetime.now()

        if writers:
            # Manually specified writers
            self.new_writers = writers
            logger.info(f"Using manually specified writers: {writers}")
        elif self.config.auto_detect:
            # Auto-detect new writers
            self.new_writers = detect_new_writers()
            logger.info(f"Auto-detected new writers: {self.new_writers}")
        else:
            raise ValueError("No writers specified and auto-detect is disabled")
        
        if not self.new_writers:
            logger.warning("No new writers found to process")

        report['steps']['detect_writers'] = {
            'status': 'completed',
            'duration_seconds': (datetime.now() - step_start).total_seconds(),
            'new_writers': self.new_writers,
            'method': 'manual' if writers else 'auto-detect'
        }

        if dry_run:
            logger.info(f"[DRY RUN] Would process {len(self.new_writers)} writers: {self.new_writers}")

    def _step_create_version(self, dry_run: bool, report: Dict[str, Any]) -> None:
        """ Step 2: Create new version or use existing """
        logger.info("Step 2: Creating new version...")
        step_start = datetime.now()

        # Get latest version
        latest_version = get_latest_version_number()
        if not dry_run:
            # Get latest version and create new one
            version_path = create_new_version(writers=self.new_writers, description='Auto-generated version')
            self.current_version = int(version_path.name[1:])
            logger.info(f"Created version: {self.current_version}")

            # Copy existing data from previous version
            if latest_version and latest_version != f"v{self.current_version}":
                logger.info(f"Copying existing data from {latest_version} to v{self.current_version}")
                copy_success = copy_existing_data(latest_version, f"v{self.current_version}")
                if copy_success:
                    logger.info("Successfully copied existing data from previous version")
                else:
                    logger.warning("Failed to copy existing data from previous version")
        else:
            # Simulate version creation
            if latest_version:
                version_num = int(latest_version[1:])
                self.current_version = version_num + 1
            else:
                self.current_version = 1
            logger.info(f"[DRY RUN] Would create version: v{self.current_version}")

        # Update report
        report['steps']['create_version'] = {
            'status': 'completed',
            'duration_seconds': (datetime.now() - step_start).total_seconds(),
            'version_number': self.current_version,
            'version_path': str(self.config.trocr_ready_dir / f"v{self.current_version}")
        }

    def _step_run_segmentation(self, dry_run: bool, report: Dict[str, Any]) -> None:
        """ Step 3: Run segmentation for new writers only """
        logger.info("Step 3: Running segmentation for new writers...")
        step_start = datetime.now()

        if not self.new_writers:
            logger.info("No new writers to segment, skipping...")
            report['steps']['segmentation'] = {
                'status': 'skipped',
                'duration_seconds': (datetime.now() - step_start).total_seconds(),
                'reason': 'no_new_writers'
            }
            return
        
        version_dir = self.config.trocr_ready_dir / f"v{self.current_version}"

        if not dry_run:
            # Convert List[str] to List[Dict]
            # Note: w is already clean (writer01), but original folder might be writer_01
            writers_data = []
            for w in self.new_writers:
                # Try to find the actual folder (could be writer01 or writer_01)
                original_folder = None
                for folder_name in [w, f'writer_{w[6:]}']:  # Try writer01, then writer_01
                    potential_path = self.config.originals_dir / folder_name
                    if potential_path.exists():
                        original_folder = folder_name
                        break
                
                if original_folder is None:
                    logger.warning(f"Could not find folder for writer {w}")
                    continue
                    
                writers_data.append({'writer_id': w, 'path': self.config.originals_dir / original_folder})

            # Run segmentation for new writers
            segmentation_report = run_segmentation_for_multiple_writers(
                writers_data=writers_data,
                output_dir=version_dir,
                enable_references=True,
                enable_visualization=True
            )

            total_images = sum(result.get('total_images', 0) for result in segmentation_report.values()) ##
            logger.info(f"Segmentation completed. Total images: {total_images}")
        else:
            logger.info(f"[DRY RUN] Would segment {len(self.new_writers)}")
            total_images = len(self.new_writers) * 100 #Estimate

        # Update report
        report['steps']['segmentation'] = {
            'status': 'completed',
            'duration_seconds': (datetime.now() - step_start).total_seconds(),
            'processed_writers': self.new_writers,
            'total_images': total_images if not dry_run else f"~{total_images} (estimated)"
        }

    
    def _step_generate_synthetic_data(self, dry_run: bool, report: dict) -> None:
        """ Step 3.4: Generate synthetic data if not already present """
        version_dir = self.config.trocr_ready_dir / f"v{self.current_version}"
        images_dir = version_dir / "images"

        # Check if synthetic images exist (filename begins with 'synthetic_')
        already_exists = any(str(f).startswith('synthetic_') for f in images_dir.glob('*.jpg'))
        if not already_exists and not dry_run:
            logger.info(f"Generating synthetic images in {images_dir} ...")
            generate_synthetic_data(images_dir)
        else:
            logger.info("Synthetic images exist or dry_run, skipping generation")
        
        report['steps']['synthetic_data'] = {
            'status': 'completed' if not dry_run else 'skipped',
            'image_dir': str(images_dir)
        }

    def _step_interactive_quality_control(self, dry_run: bool, report: Dict[str, Any]) -> None:
        """ Step 3.5: Interactive Quality Control for problematic images """
        logger.info("Step 3.5: Interactive Quality Control...")
        step_start = datetime.now()

        if dry_run:
            logger.info("[DRY RUN] Would run interactive QC")
            report['steps']['quality_control'] = {
            'status': 'skipped',
            'reason': 'dry_run',
            'duration_seconds': (datetime.now() - step_start).total_seconds()
            }
            return

        version_dir = self.config.trocr_ready_dir / f"v{self.current_version}"
        total_removed = 0
        original_count = count_total_files(version_dir)

        logger.info("===SEGMENTATION COMPLETE===")
        logger.info(f"Created {original_count} segmented images")

        while True:
            user_input = input("\nInput writer:word pairs to remove (comma separated) or None to continue:\n>>> ").strip()

            writer_word_pairs = parse_writer_word_input(user_input)
            if not writer_word_pairs:
                break

            # Process removal with verbose=True for immediate feedback
            removed_files, failed_pairs, messages = remove_files_batch(
                writer_word_pairs, self.config.trocr_ready_dir, dry_run=False, verbose=True,
                specific_version=self.current_version
            )

            # Count actual unique files removed (avoid duplicates in counting)
            batch_removed = len(set(str(f) for f in removed_files))
            total_removed += batch_removed

            # Show updated statistics
            current_count = count_total_files(version_dir)  # Re-count actual remaining files
            logger.info(f"Statistics: {current_count} files remaining ({original_count - current_count} removed total)")

            # Show failures
            if failed_pairs:
                failed_strings = [f"{writer}:{word}" for writer, word in failed_pairs]
                logger.warning(f"Not found: {', '.join(failed_strings)}")

        logger.info("===PROCEEDING TO FINALIZATION===")

        report['steps']['quality_control'] = {
            'status': 'completed',
            'duration_seconds': (datetime.now() - step_start).total_seconds(),
            'files_removed': total_removed,
            'files_remaining': original_count - total_removed,
            'original_count': original_count
        }

    def _step_create_annotations(self, dry_run: bool, report: Dict[str, Any]) -> None:
        """ Step 4: Create complrehensive annotations for all data """
        logger.info("Step 4: Creating annotations")
        step_start = datetime.now()

        version_dir = self.config.trocr_ready_dir / f"v{self.current_version}"
        images_dir = version_dir / "images"

        if not dry_run:
            # Create annotations from all images in version directory
            annotations_file = create_annotations_for_version(
                images_dir=images_dir,
                output_dir=version_dir,
                metadata_dir=None # Use images_dir parent as fallback
            )

            # Load and count annotations
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            total_annotations = len(annotations)
            unique_words = len(set(ann['ground_truth_text'] for ann in annotations))
            unique_writers = len(set(ann['writer_id'] for ann in annotations))

            logger.info(f"Created {total_annotations} annotations ({unique_words} unique words, {unique_writers} writers)")
        else:
            logger.info(f"[DRY RUN] Would create annotations for images in {images_dir}")
            total_annotations = 'estimated_from_images'
            unique_words = 'estimated'
            unique_writers = len(self.new_writers)

        # Update report
        report['steps']['create_annotations'] = {
            'status': 'completed',
            'duration_seconds': (datetime.now() - step_start).total_seconds(),
            'total_annotations': total_annotations,
            'unique_words': unique_words,
            'unique_writers': unique_writers,
            'annotations_file': str(version_dir / "annotations.json") if not dry_run else "would_create"
        }
    def _step_apply_augmentation(self, dry_run: bool, report: Dict[str, Any]) -> None:
        """ Step 5: Apply data augmentation (optional) """
        logger.info("Step 5: Applying data augmentation")
        step_start = datetime.now()

        version_dir = self.config.trocr_ready_dir / f"v{self.current_version}"

        if not dry_run:
            # Apply augmentation to all images in version
            aug_annotations_path, aug_config_path = create_augmented_training_data(
                version_dir=version_dir,
                config=None, # Use default augmentation config
                augmentations_per_image=self.config.augmentations_per_image
            )

            # Load and count augmented annotations
            with open(aug_annotations_path, 'r', encoding='utf-8') as f:
                aug_annotations = json.load(f)

            original_count = len([ann for ann in aug_annotations if ann.get('category') != 'word_augmented'])
            augmented_count = len([ann for ann in aug_annotations if ann.get('category') == 'word_augmented'])

            logger.info(f"Augmentation completed. Original: {original_count}, Augmented: {augmented_count}")
        else:
            logger.info(f"[DRY RUN] Would create {self.config.augmentations_per_image} augmented versions per image")
            original_count = 'estimated'
            augmented_count = f"estimated * {self.config.augmentations_per_image}"

        # Update report
        report['steps']['augmentation'] = {
            'status': 'completed',
            'duration_seconds': (datetime.now() - step_start).total_seconds(),
            'augmentations_per_image': self.config.augmentations_per_image,
            'original_annotations': original_count,
            'augmented_annotations': augmented_count,
            'augmented_annotations_file': str(version_dir / "annotations_augmented.json") if not dry_run else "would_create"
        }

    def _step_create_splits(self, dry_run: bool, report: Dict[str, Any]) -> None:
        """ Step 6: Create stratified train/val/test splits """
        logger.info("Step 6: Creating dataset splits")
        step_start = datetime.now()

        version_dir = self.config.trocr_ready_dir / f"v{self.current_version}"

        # Use augmented annotations if available, otherwise use regular annotations
        if self.config.apply_augmentation:
            annotations_file = version_dir / "annotations_augmented.json"
        else:
            annotations_file = version_dir / "annotations.json"

        if not dry_run:
            # Create stratified splits
            split_files = create_dataset_splits(
                annotations_path=annotations_file,
                output_dir=version_dir,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                test_ratio=self.config.test_ratio,
                random_state=42,
                use_augmented=True
            )

            # Count samples in each split
            split_counts = {}
            for split_name, split_file in split_files.items():
                with open(split_file, 'r', encoding='utf-8') as f:
                    split_counts[split_name] = sum(1 for line in f if line.strip())

            total_samples = sum(split_counts.values())
            logger.info(f"Dataset splits created. Train {split_counts.get('train', 0)}, "
                        f"Val: {split_counts.get('val', 0)}, Test: {split_counts.get('test', 0)}")
        else:
            logger.info(f"[DRY RUN] Would create train/val/test splits from {annotations_file}")
            split_counts = {'train': 'estimated', 'val': 'estimated', 'test': 'estimated'}
            total_samples = 'estimated'

        # Update report
        report['steps']['create_splits'] = {
            'status': 'completed',
            'duration_seconds': (datetime.now() - step_start).total_seconds(),
            'annotations_source': str(annotations_file),
            'split_ratios': {
                'train': self.config.train_ratio,
                'val': self.config.val_ratio, 
                'test': self.config.test_ratio
            },
            'split_counts': split_counts,
            'total_samples': total_samples
        }

    def _step_validate_and_cleanup(self, dry_run: bool, report: Dict[str, Any]) -> None:
        """ Step 7: Validate pipeline results and cleanup old versions """
        logger.info("Step 7: Validating results and cleanup...")
        step_start = datetime.now()

        version_dir = self.config.trocr_ready_dir / f"v{self.current_version}"
        validation_results = {}

        if not dry_run:
            # Validate files exist
            expected_files = [
                'annotations.json',
                'gt_train.txt',
                'gt_val.txt',
                'gt_test.txt'
            ]

            if self.config.apply_augmentation:
                expected_files.extend([
                    'annotations_augmented.json',
                    'augmentation_config.json'
                ])

            missing_files = []
            for file_name in expected_files:
                file_path = version_dir / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            validation_results['missing_files'] = missing_files
            validation_results['all_files_present'] = len(missing_files) == 0

            # Cleanup old versions (keep only latest N versions)
            if self.config.keep_versions > 0:
                self._cleanup_old_versions()
                validation_results['cleanup_performed'] = True
            
            if missing_files:
                logger.warning(f"Validation failed: Missing files {missing_files}")
            else:
                logger.info("Validation passed: All expected files present")
            
        else:
            logger.info(f"[DRY RUN] Would validate files in {version_dir}")
            logger.info(f"[DRY RUN] Would cleanup old versions (keep {self.config.keep_versions})")
            validation_results['dry_run'] = True

        # Update report
        report['steps']['validate_cleanup'] = {
            'status': 'completed',
            'duration_seconds': (datetime.now() - step_start).total_seconds(),
            'validation_results': validation_results,
            'version_dir': str(version_dir)
        }

    def _cleanup_old_versions(self) -> None:
        """ Remove old versions, keep only the latest N versions """
        if not self.config.trocr_ready_dir.exists():
            return
        
        # Find all version directories
        version_dirs = []
        for path in self.config.trocr_ready_dir.iterdir():
            if path.is_dir() and path.name.startswith('v') and path.name[1:].isdigit():
                version_num = int(path.name[1:])
                version_dirs.append((version_num, path))

        version_dirs.sort(key=lambda x:x[0], reverse=True)
        versions_to_remove = version_dirs[self.config.keep_versions:]

        for version_num, version_path in versions_to_remove:
            logger.info(f"Removing old version: v{version_num}")
            shutil.rmtree(version_path)

def main():
    """ Command line interface for pipeline execution """
    parser = argparse.ArgumentParser(description="Complete TrOCR data pipeline")
    parser.add_argument('--auto-detect', action='store_true', default=True,
                        help='Auto-detect new writers (default: True)')
    parser.add_argument('--writers', type=str, 
                        help='Comma-separated list of specific writers (e.g., writer_01,writer_02)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Skip data augmentation step')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without executing')
    parser.add_argument('--keep-versions', type=int, default=3,
                        help='Number of versions to keep (default: 3)')
    
    args = parser.parse_args()

    # Setup configuration
    config = PipelineConfig()
    config.apply_augmentation = not args.no_augmentation
    config.keep_versions = args.keep_versions

    # Parse writers if specified
    writers = None
    if args.writers:
        writers = [w.strip() for w in args.writers.split(',')]
        config.auto_detect = False

    # Run pipeline
    runner = PipelineRunner(config)
    try:
        report = runner.run_complete_pipeline(writers=writers, dry_run=args.dry_run)

        if config.save_reports:
            report_file = Path('logs') / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Pipeline report saved to: {report_file}")

        print("Pipeline completed successfully!")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())