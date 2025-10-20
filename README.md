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
│   ├── trocr_ready_data/        # Complete TrOCR-ready datasets by version
│   │   ├── v1/                  # Dataset version 1
│   │   │   ├── images/          # 384x384 segmented word images
│   │   │   ├── gt_train.txt     # Training data (tab-separated)
│   │   │   ├── gt_val.txt       # Validation data
│   │   │   ├── gt_test.txt      # Test data
│   │   │   └── ...              # Metadata and config files
│   │   └── v2/                  # Dataset version 2 (incremental updates)
│   ├── segmented_words/         # Legacy: Individual writer segmentation (deprecated)
│   ├── splits/                  # Legacy: Old split format (deprecated)  
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
│   ├── training/                # TrOCR model training pipeline
│   │   ├── train_model.py          # Main training script with Azure ML support
│   │   ├── dataset_loader.py       # Swedish handwriting dataset loader
│   │   └── evaluation/
│   │       └── metrics.py          # Comprehensive Swedish OCR metrics
│   └── evaluation/              # Model evaluation and testing scripts
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

### Phase 2: Complete Data Processing Pipeline
- Process all data from raw scans to TrOCR-ready format using the integrated orchestrator
```bash
cd /home/fendraq/wsl_projects/swedish_handwritten_ocr

# Complete pipeline (recommended)
python -m scripts.data_processing.orchestrator.main --auto-detect

# Alternative: specific writers
python -m scripts.data_processing.orchestrator.main --writers writer_01,writer_02

# Dry run to preview actions
python -m scripts.data_processing.orchestrator.main --auto-detect --dry-run
```

### Phase 2.5: Manual Image Quality Control (Critical Step)
**Important**: After segmentation but before final training dataset creation, perform manual quality control on segmented images:

#### Manual Editing Workflow:
1. **Location**: `dataset/trocr_ready_data/vX/images/` (segmented word images)
2. **Timing**: After segmentation, before annotation/training
3. **Focus areas**:
   - **Kladd/överstrykningar**: Remove crossed-out letters/words
   - **Följande linjer**: Remove template lines that interfere with text
   - **Otydlig text**: Flag or remove illegible samples
   - **Överskriven text**: Keep final intended character (e.g., if 'l' is overwritten with 'r', ground truth should be 'r')

#### Quality Control Guidelines:
- **Edit segmented images**: Work on individual word images, NOT full-page scans
- **Preserve segmentation**: Don't modify image dimensions (384x384)
- **Ground truth consistency**: Final text should match visual intent
- **Document changes**: Keep track of edited images for quality control
- **Backup originals**: Maintain copies of unedited segments

#### Recommended tools:
- GIMP, Photoshop, or any image editor capable of precise editing
- Batch processing tools for consistent operations

**Rationale**: Editing segmented images preserves segmentation quality while providing clean training data for TrOCR, minimizing risk of segmentation errors that could occur from editing full-page scans.

#### Automated Problematic Image Removal:
For systematic removal of problematic segmented images (including augmented versions):

```bash
cd /home/fendraq/wsl_projects/swedish_handwritten_ocr

# List all versions of a problematic image (original + augmented)
python -m scripts.data_processing.remove_problematic_images --list-pattern writer01_page08_143_EVIGHET

# Dry-run to see what would be removed
python -m scripts.data_processing.remove_problematic_images --remove writer01_page08_143_EVIGHET --dry-run

# Remove problematic image and all augmented versions
python -m scripts.data_processing.remove_problematic_images --remove writer01_page08_143_EVIGHET

# Interactive mode for handling multiple problematic images
python -m scripts.data_processing.remove_problematic_images --interactive
```

**Features:**
- **Finds all versions**: Automatically locates original + all augmented versions (`_aug_01.jpg`, `_aug_02.jpg`, etc.)
- **Multi-version support**: Works across all dataset versions (v1, v2, v3, etc.)
- **Safe operation**: Dry-run mode to preview changes before execution
- **Interactive workflow**: Step-by-step guidance for quality control decisions
- **Pattern matching**: Handles base filename without extensions or augmentation suffixes

**Usage scenarios:**
- Remove corrupted/illegible images
- Delete images with excessive template line interference
- Clean up badly segmented words (cut-off text, wrong boundaries)
- Maintain dataset quality before training

### Phase 2.1: Individual Module Testing (Advanced)
- Run individual pipeline components for debugging
```bash
cd /home/fendraq/wsl_projects/swedish_handwritten_ocr

# Legacy individual segmentation (for single writer testing)
python -m scripts.data_processing.image_segmentation.segment_images \
    --metadata "docs/data_collection/generated_templates/complete_template_metadata.json" \
    --images "dataset/originals/writer_01" \
    --output "dataset/segmented_words" \
    --writer-id "writer_01" \
    --visualize
```

#### Orchestrator Pipeline Features:
- **Complete automation**: Auto-detects new writers and processes end-to-end
- **TrOCR-compatible output**: Creates `gt_train.txt`, `gt_val.txt`, `gt_test.txt` files
- **Proper dataset splitting**: 70/15/15 train/val/test distribution (not 99%/0.3%/0.6%)
- **Flat output structure**: `trocr_ready_data/vX/images/` optimized for TrOCR training  
- **Clean filename format**: `writer01_page01_001_text.jpg` (underscores removed from writer_id)
- **384x384 preprocessing**: Integrated TrOCR-ready image formatting
- **Automatic annotations**: Extracts ground truth from filename metadata
- **Data augmentation**: Optional rotation, blur, brightness/contrast variations
- **Version management**: Incremental dataset versions with cleanup
- **Validation**: Ensures all expected files are created correctly

#### Complete Pipeline Steps:
1. **data_detector.py**: Scans originals/ directory for new writers to process
2. **version_manager.py**: Creates new dataset version (v1, v2, etc.) with metadata
3. **segmentation_runner.py**: Converts raw scans to 384x384 segmented word images
4. **Manual Quality Control**: Edit segmented images to remove kladd, lines, and ensure text clarity
5. **annotation_creator.py**: Extracts ground truth from filenames → `annotations.json`
6. **augmentation_manager.py**: Applies optional data augmentation for training robustness
7. **dataset_splitter.py**: Creates TrOCR-compatible train/val/test splits → `gt_*.txt` files
8. **Validation & Cleanup**: Verifies output files and removes old dataset versions

**Note**: Step 4 (Manual Quality Control) should be performed on segmented images before proceeding to annotation and dataset splitting.

#### Output Structure:
```
trocr_ready_data/
├── v1/
│   ├── images/                    # All 384x384 segmented images (JPG)
│   │   ├── writer01_page01_001_Åsa.jpg
│   │   ├── writer01_page01_002_huvudgång.jpg
│   │   └── ...
│   ├── images_augmented/          # Augmented training images (optional)
│   ├── annotations.json           # Ground truth metadata (JSON format)
│   ├── annotations_augmented.json # Combined original + augmented annotations
│   ├── gt_train.txt              # Training split (TrOCR tab-separated format)
│   ├── gt_val.txt                # Validation split (15% of data)
│   ├── gt_test.txt               # Test split (15% of data)  
│   ├── metadata.json             # Version metadata (writers, counts, etc.)
│   ├── augmentation_config.json  # Augmentation parameters
│   └── segmentation_summary.json # Segmentation statistics
```

**TrOCR Format Example (gt_train.txt):**
```
images/writer01_page01_001_Åsa.jpg	Åsa
images/writer01_page01_002_huvudgång.jpg	huvudgång
images/writer01_page02_025_KAPELL.jpg	KAPELL
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

# Main orchestrator pipeline (recommended)
python -m scripts.data_processing.orchestrator.main --auto-detect

# Alternative command options:
python -m scripts.data_processing.orchestrator.main --writers writer_01,writer_02 --no-augmentation
python -m scripts.data_processing.orchestrator.main --auto-detect --dry-run --keep-versions 5

# Individual modules (for development/debugging)
python -m scripts.data_processing.template_generator.generate_templates
python -m scripts.data_processing.image_segmentation.segment_images [options]
```

**Orchestrator Command Options:**
- `--auto-detect`: Automatically detect new writers in dataset/originals/
- `--writers writer_01,writer_02`: Process specific writers only  
- `--no-augmentation`: Skip data augmentation step
- `--dry-run`: Preview actions without executing
- `--keep-versions N`: Keep N most recent dataset versions (default: 3)

**Path Management:**
The project uses centralized path configuration in `config/paths.py`. All file paths are relative to the project root, ensuring compatibility with Azure ML and other deployment environments.

### Phase 3: Model Training

The project includes a complete TrOCR fine-tuning pipeline with Azure ML compatibility, comprehensive metrics evaluation, and intelligent path management.

#### Training Pipeline Features:
- **HuggingFace Seq2SeqTrainer**: Optimized training loop with automatic mixed-precision and gradient accumulation
- **Custom TrOCR Data Collator**: Handles pixel_values (images) + labels (text tokens) format
- **Swedish-specific metrics**: Character Error Rate (CER), Word Error Rate (WER), BLEU evaluation
- **VisionEncoderDecoderModel**: Based on Microsoft documentation
- **WandB integration**: Experiment tracking and performance monitoring
- **Comprehensive evaluation**: Multi-metric evaluation optimized for Swedish handwriting recognition

#### Quick Start Training:

```bash
cd /home/fendraq/wsl_projects/swedish_handwritten_ocr

# Dry run with small dataset (10 samples)
python -m scripts.training.train_model --dry_run --epochs 1

# Full training with WandB logging
python -m scripts.training.train_model --epochs 10 --wandb

# Custom training parameters
python -m scripts.training.train_model --batch_size 8 --epochs 5 --learning_rate 1e-5
```

#### Training Validation Results:
- ✅ **Data loading**: Successfully loads gt_train.txt and gt_val.txt
- ✅ **Custom collator**: TrOCRDataCollator handles vision-to-text data correctly  
- ✅ **Model training**: VisionEncoderDecoderModel trains without ValueError
- ✅ **Model saving**: Trained models saved to `models/trocr-swedish-handwriting-v3-*/final_model/`
- ✅ **Metrics integration**: CER, WER, BLEU metrics configured for evaluation
- ✅ **No critical warnings**: Fixed image processor and generation config warnings

#### Training Configuration (Validated):
- **Base model**: microsoft/trocr-base-handwritten (✅ Working)
- **Architecture**: VisionEncoderDecoderModel (updated from TrOCRForCausalLM)
- **Batch size**: 16 (with gradient accumulation for effective batch size of 32)
- **Learning rate**: 5e-5 with warmup
- **Mixed precision**: FP16 enabled with use_fast=True for TrOCRProcessor
- **Evaluation strategy**: Every 200 steps with CER-based best model selection
- **Data collation**: Custom TrOCRDataCollator for pixel_values + labels format

#### Trained Models Available:
Current trained models ready for evaluation:
```
models/
├── trocr-swedish-handwriting-v3-20251020_091130/
│   ├── final_model/              # ✅ Complete trained model (1.3GB)
│   │   ├── model.safetensors     # Model weights
│   │   ├── config.json           # Model configuration  
│   │   ├── tokenizer.json        # Swedish tokenizer
│   │   └── preprocessor_config.json
│   └── checkpoint-1/             # Training checkpoint
└── (other training runs...)
```

#### Azure ML Deployment (Ready):

The training pipeline automatically detects Azure ML environments and adapts paths accordingly:

```bash
# Local development (automatic detection)
python -m scripts.training.train_model --epochs 10

# Azure ML (automatic detection via environment variables)
# No code changes needed - paths.py handles environment detection
```

**Azure ML Environment Variables Detected:**
- `AZUREML_RUN_ID`
- `AZUREML_EXPERIMENT_ID` 
- `AZUREML_DATAREFERENCE_data`

#### Metrics and Evaluation:

The training pipeline includes comprehensive Swedish handwriting metrics:

1. **Character Error Rate (CER)**: Core metric for OCR evaluation
2. **Word Error Rate (WER)**: Word-level accuracy measurement
3. **BLEU Score**: Text generation quality assessment
4. **Swedish Character Accuracy**: Specialized metric for å, ä, ö recognition
5. **Exact Match Accuracy**: Perfect prediction percentage

**Metrics Output Example:**
```
eval_cer: 0.0234
eval_wer: 0.1250  
eval_bleu: 0.8567
eval_swedish_chars: 0.9234
eval_exact_match: 0.7890
```

#### Model Output:
Trained models are saved with version information and timestamps:
```
models/
└── trocr-swedish-handwriting-v3-20241017_143052/
    ├── final_model/          # Complete model and tokenizer
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── tokenizer files
    └── checkpoints/          # Training checkpoints
```

#### Path Management:
The training system uses intelligent path detection from `config/paths.py`:
- **Local development**: Uses dataset versions from `dataset/trocr_ready_data/`
- **Azure ML**: Automatically detects mounted data and adjusts paths
- **Version management**: Always uses latest dataset version automatically

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
  - Tab-separated gt_*.txt format following Microsoft TrOCR standard
  - Proper 70/15/15 train/validation/test distribution for robust model training
  - Clean filename format (writer01 vs writer_01) for reliable parsing
- **Quality assurance**:
  - Visualization tools for debugging marker detection
  - Consistent font rendering across all template text
  - Optimized line thickness (0.5pt) for clean segmentation
- **Production ready**: Configurable pipeline suitable for large-scale dataset generation
- **Complete TrOCR pipeline**: Production-ready fine-tuning with Azure ML compatibility
- **Intelligent path management**: Automatic environment detection for local and cloud deployment
- **Advanced metrics evaluation**: Swedish-specific accuracy measurements for å, ä, ö characters
