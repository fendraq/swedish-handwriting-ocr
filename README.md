# Swedish Handwriting OCR Project

A project for training a TrOCR model for Swedish handwritten text recognition, specifically designed for form processing applications.

## Project Structure

```
swedish_handwritten_ocr/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml              # Configuration file for training and dataset
├── dataset/
│   ├── originals/               # Original scanned documents (JPG format)
│   │   ├── annotations/         # Metadata and labels
│   │   ├── writer_01/
│   │   ├── writer_02/
│   │   └── ...
│   ├── segmented_words/         # Segmented word images organized by writer
│   │   ├── writer_001/
│   │   ├── writer_002/
│   │   └── ...
│   ├── preprocessed/            # 384x384 processed images for TrOCR
│   ├── splits/                  # Train/validation/test splits
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── azure_ready/             # Azure ML formatted data
│   └── segmented_words_visualizations/  # Debug visualizations
├── scripts/
│   ├── data_processing/         # Data processing and orchestration
│   │   ├── orchestrator/        # Main pipeline orchestration
│   │   │   ├── main.py             # Complete data processing pipeline
│   │   │   ├── data_detector.py    # Detect new writers in dataset
│   │   │   ├── version_manager.py  # Handle dataset versioning
│   │   │   ├── segmentation_runner.py # Segmentation pipeline wrapper
│   │   │   ├── annotation_creator.py  # Ground truth annotation generator
│   │   │   ├── dataset_splitter.py    # Stratified train/val/test splitting
│   │   │   └── augmentation_manager.py # Data augmentation pipeline
│   │   ├── template_generator/  # PDF template generation
│   │   │   └── generate_templates.py # Generate handwriting templates
│   │   ├── image_segmentation/  # Image processing and segmentation
│   │   │   └── segment_images.py # Segment scanned images
│   │   └── data_preparation/    # Data formatting utilities
│   ├── training/                # Model training scripts
│   └── evaluation/              # Evaluation scripts
├── models/
│   ├── checkpoints/             # Training checkpoints
│   └── final/                   # Final models
├── docs/                        # Documentation
│   └── data_collection/         # Data collection templates and resources
│       ├── generated_templates/ # PDF templates with metadata
│       └── word_collections/    # Swedish word lists and vocabularies
└── logs/                        # Training and evaluation logs
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Generation

The project uses word lists in `docs/data_collection/word_collections/` containing categorized Swedish words optimized for handwriting recognition, including funeral/burial terminology with focus on Swedish characters (å, ä, ö).

### Phase 1: Data Collection
- Create templates based on the word list
```bash
cd /home/fendraq/wsl_projects/swedish_handwritten_ocr

python -m scripts.data_processing.template_generator.generate_templates
```
- Templates include:
    - Instruction text: "Skanna in i jpg-format och skriv innanför linjerna"
    - Page numbers for correct scanning order
    - Reference markers for automatic coordinate transformation
    - Optimized line thickness (0.5 pt) for minimal margin requirements
- Print templates and distribute to 10-20 different writers
- Important: Scan completed forms as JPG files in correct page order (Sida 1, Sida 2, etc.)

### Phase 2: Data Processing
- Segment scanned documents into individual words
```bash
cd /home/fendraq/wsl_projects/swedish_handwritten_ocr

python -m scripts.data_processing.image_segmentation.segment_images --metadata "docs/data_collection/generated_templates/complete_template_metadata.json" --images "dataset/originals/writer_001" --output "dataset/segmented_words" --writer-id "writer_001" --visualize
```

### Phase 2.1: Complete Data Pipeline (Recommended)
- Process all data from raw scans to TrOCR-ready format
```bash
cd /home/fendraq/wsl_projects/swedish_handwritten_ocr

python -m scripts.data_processing.orchestrator.main --auto-detect
```

### Phase 2.2: TrOCR-Ready Data Generation
- Automated orchestrator pipeline for TrOCR training data
```bash
python -m scripts.data_processing.orchestrator.segmentation_runner
python -m scripts.data_processing.orchestrator.annotation_creator  
python -m scripts.data_processing.orchestrator.dataset_splitter
python -m scripts.data_processing.orchestrator.augmentation_manager
```

#### New Orchestrator Features:
- **Flat output structure**: `trocr_ready_data/vX/images/` instead of category folders
- **384x384 preprocessing**: Integrated during segmentation
- **Automatic annotations**: Creates `annotations.json` from filename metadata
- **Stratified dataset splitting**: 70/15/15 train/val/test splits with writer balancing
- **Data augmentation**: Rotation (±5°), blur (σ 0.3-0.8), brightness/contrast (±15%)
- **Filename format**: `{writer_id}_{page}_{word_id}_{text}.jpg`
- **Version management**: Incremental dataset versions

#### Complete Pipeline Modules:
1. **segmentation_runner.py**: Converts raw scans to 384x384 segmented words
2. **annotation_creator.py**: Extracts ground truth from filenames → `annotations.json`
3. **dataset_splitter.py**: Creates stratified train/val/test splits → JSONL files
4. **augmentation_manager.py**: Applies data augmentation for training robustness

#### Output Structure:
```
trocr_ready_data/
├── v1/
│   ├── images/                    # All 384x384 segmented images  
│   ├── images_augmented/          # Augmented training images
│   ├── annotations.json           # Ground truth annotations
│   ├── annotations_augmented.json # Combined original + augmented annotations
│   ├── train.jsonl               # Training split (TrOCR format)
│   ├── val.jsonl                 # Validation split
│   ├── test.jsonl                # Test split
│   └── augmentation_config.json  # Augmentation parameters
```
#### Segmentation Features:
- **Dynamic image analysis**: Automatically detects DPI and adjusts parameters accordingly
- **Reference marker detection**: Uses circular markers for precise coordinate transformation
- **Adaptive parameters**: Automatically scales detection based on actual image resolution (200 DPI vs 300 DPI)
- **Smart margin control**: 6-pixel inward margin to remove border artifacts from segmented words
- **Fallback mode**: Functions with simple coordinate conversion when markers aren't detected
- **Visualization support**: Generate debug images showing detected markers and segmentation regions

#### Segmentation Options:
- `--use-references`: Enable reference marker detection (default: True)
- `--no-references`: Disable markers, use simple coordinate conversion
- `--visualize`: Generate visualization images for debugging
- `--viz-output`: Custom directory for visualization output

#### File Organization:
- Segmented images organized by writer: `dataset/segmented_words/writer_001/`
- Each image named: `{category}_{word_id}_{writer_id}.jpg`
- Visualization images: `dataset/segmented_words_visualizations/`

## Important Usage Notes

**Command Execution:**
All scripts must be run from the project root directory using Python module syntax:

```bash
# Always run from project root
cd /home/fendraq/wsl_projects/swedish_handwritten_ocr

# Main orchestrator (recommended)
python -m scripts.data_processing.orchestrator.main --auto-detect

# Individual pipeline modules (for debugging)
python -m scripts.data_processing.template_generator.generate_templates
python -m scripts.data_processing.image_segmentation.segment_images [options]
```

**Path Management:**
The project uses centralized path configuration in `config/paths.py`. All file paths are relative to the project root, ensuring compatibility with Azure ML and other deployment environments.

### Phase 3: Model Training
- Fine-tune TrOCR on Swedish handwriting data
- Evaluate performance  
- Optimize hyperparameters

## Features

- **Comprehensive Swedish vocabulary**: 150+ categorized words covering Swedish characters (å, ä, ö), names, places, dates, and funeral/burial terminology
- **Intelligent template generation**: PDF templates with reference markers, page numbers, and scanning instructions
- **Advanced image segmentation**: 
  - Automatic DPI detection and parameter adaptation
  - Reference marker detection for precise coordinate mapping
  - Dynamic margin adjustment (6px inward) to remove border artifacts
  - Fallback coordinate transformation when markers aren't available
- **Intelligent data pipeline**:
  - Automated annotation generation from filename metadata
  - Stratified dataset splitting ensuring writer balance and word representation
  - Configurable data augmentation (rotation, blur, brightness/contrast)
  - Version-controlled dataset management with incremental updates
- **TrOCR optimization**:
  - 384x384 preprocessing integrated in segmentation pipeline
  - JSONL output format compatible with HuggingFace datasets
  - Separate augmented data for training robustness
- **Quality assurance**:
  - Visualization tools for debugging marker detection
  - Consistent font rendering across all template text
  - Optimized line thickness (0.5pt) for clean segmentation
- **Production ready**: Configurable pipeline suitable for large-scale dataset generation
- **TrOCR integration**: Fine-tuning pipeline using Microsoft's TrOCR model (upcoming)
