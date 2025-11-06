# Swedish Handwriting OCR

A production-ready system for training and deploying TrOCR models for Swedish handwritten text recognition. The system provides comprehensive tools for data collection, preprocessing, model training, and evaluation with cloud platform compatibility.

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
│   │   ├── train_model.py          # Main training script with cloud support
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

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB RAM minimum

### Installation
```bash
# Clone and setup environment
git clone <repository-url>
cd swedish_handwritten_ocr
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Basic Usage
```bash
# Generate training data from scanned forms (LEGACY: word-level only, will be updated for line-level)
python -m scripts.data_processing.orchestrator.main --auto-detect

# Train TrOCR model on Swedish handwriting
python -m scripts.training.train_model --epochs 30 --wandb

# Evaluate trained model performance
python -m scripts.evaluation.evaluate_model
```

## System Components

### Data Processing Pipeline
**Location**: `scripts/data_processing/`

**Core Components**:
- **Template Generation**: Creates handwriting collection forms with reference markers (LEGACY: word-level templates, will support line-level)
- **Image Segmentation**: Automatically processes scanned documents (LEGACY: word segmentation, will support line segmentation)
- **Quality Control**: Interactive and automated tools for dataset curation
- **Data Orchestration**: End-to-end pipeline from raw scans to TrOCR-ready datasets (LEGACY: will be rebuilt for line-level processing)

### Training Pipeline
**Location**: `scripts/training/`

**Features**:
- **HuggingFace Integration**: Seq2SeqTrainer with automatic mixed-precision training
- **Swedish Optimization**: Custom metrics and tokenization for Swedish characters (å, ä, ö)
- **Cloud Platform Support**: Automatic environment detection for cloud deployment
- **Model Persistence**: Complete model saving with tokenizer and configuration

### Evaluation System
**Location**: `scripts/evaluation/`

**Capabilities**:
- **Multi-metric Assessment**: CER, WER, BLEU, and Swedish-specific accuracy
- **Environment Awareness**: Adapts between local development and production evaluation
- **Ground Truth Validation**: Comprehensive test split evaluation with proper data isolation
- **Error Analysis**: Detailed performance analysis and failure pattern identification

## Data Collection and Processing (LEGACY - Word-Level)

> **NOTE**: Current orchestrator processes word-level data. Future versions will support line-level text collection and processing for more realistic training data.

The system uses curated Swedish word collections in `docs/data_collection/word_collections/` containing categorized vocabulary optimized for handwriting recognition, including specialized terminology with focus on Swedish characters (å, ä, ö).

### Phase 1: Template Generation and Data Collection (LEGACY - Word-Level)
**Generate handwriting collection templates:**
```bash
# LEGACY: Word-level templates only
python -m scripts.data_processing.template_generator.generate_templates
```

**Template features:**
- Instruction text: "Skanna in i jpg-format och skriv innanför linjerna"
- Page numbers for correct scanning order
- Reference markers for automatic coordinate transformation
- Optimized line thickness (0.5 pt) for minimal margin requirements

**Collection workflow:**
1. Print generated templates
2. Distribute to 10-20 different writers
3. Scan completed forms as JPG files in correct page order (Sida 1, Sida 2, etc.)

### Phase 2: Automated Data Processing (LEGACY - Word-Level)
**Run the complete processing pipeline:**
```bash
# Comprehensive pipeline (LEGACY: word-level segmentation)
python -m scripts.data_processing.orchestrator.main --auto-detect

# Process specific writers
python -m scripts.data_processing.orchestrator.main --writers writer_01,writer_02

# Preview actions without execution
python -m scripts.data_processing.orchestrator.main --auto-detect --dry-run
```

**Pipeline capabilities (LEGACY - Word-Level):**
- **Complete automation**: Auto-detects new writers and processes end-to-end
- **TrOCR-compatible output**: Creates `gt_train.txt`, `gt_val.txt`, `gt_test.txt` files
- **Proper dataset splitting**: 70/15/15 train/val/test distribution
- **Flat output structure**: `trocr_ready_data/vX/images/` optimized for TrOCR training
- **Clean filename format**: `writer01_page01_001_text.jpg` format
- **Word-level segmentation**: 384x384 individual word images (will be updated to line-level)
- **Automatic annotations**: Extracts ground truth from filename metadata
- **Data augmentation**: Optional rotation, blur, brightness/contrast variations
- **Version management**: Incremental dataset versions with cleanup
- **Validation**: Ensures all expected files are created correctly

### Phase 2.5: Quality Control (Critical Step)
After segmentation but before final training dataset creation, perform quality control on segmented images:

**Manual Editing Workflow:**
1. **Location**: `dataset/trocr_ready_data/vX/images/` (segmented word images)
2. **Timing**: After segmentation, before annotation/training
3. **Focus areas**:
   - **Crossed-out text**: Remove kladd/överstrykningar
   - **Template lines**: Remove following lines that interfere with text
   - **Illegible text**: Flag or remove unclear samples
   - **Overwritten text**: Keep final intended character

**Quality Control Guidelines:**
- **Edit segmented images**: Work on individual word images, not full-page scans
- **Preserve segmentation**: Maintain 384x384 dimensions
- **Ground truth consistency**: Final text should match visual intent
- **Document changes**: Keep track of edited images for quality control
- **Backup originals**: Maintain copies of unedited segments

**Recommended tools:** GIMP, Photoshop, or any image editor capable of precise editing

**Automated Problematic Image Removal:**
```bash
# List all versions of a problematic image (original + augmented)
python -m scripts.data_processing.remove_problematic_images --list-pattern writer01_page08_143_EVIGHET

# Dry-run to see what would be removed
python -m scripts.data_processing.remove_problematic_images --remove writer01_page08_143_EVIGHET --dry-run

# Remove problematic image and all augmented versions
python -m scripts.data_processing.remove_problematic_images --remove writer01_page08_143_EVIGHET

# Interactive mode for handling multiple problematic images
python -m scripts.data_processing.remove_problematic_images --interactive
```

**Removal features:**
- **Finds all versions**: Automatically locates original + all augmented versions
- **Multi-version support**: Works across all dataset versions (v1, v2, v3, etc.)
- **Safe operation**: Dry-run mode to preview changes before execution
- **Interactive workflow**: Step-by-step guidance for quality control decisions
- **Pattern matching**: Handles base filename without extensions or augmentation suffixes

### Advanced: Individual Module Testing

For debugging or development purposes, run individual pipeline components:

```bash
# Legacy individual segmentation (for single writer testing)
python -m scripts.data_processing.image_segmentation.segment_images \
    --metadata "docs/data_collection/generated_templates/complete_template_metadata.json" \
    --images "dataset/originals/writer_01" \
    --output "dataset/segmented_words" \
    --writer-id "writer_01" \
    --visualize
```

**Orchestrator Command Options:**
- `--auto-detect`: Automatically detect new writers in dataset/originals/
- `--writers writer_01,writer_02`: Process specific writers only
- `--no-augmentation`: Skip data augmentation step
- `--dry-run`: Preview actions without executing
- `--keep-versions N`: Keep N most recent dataset versions (default: 3)

**Pipeline Processing Steps (LEGACY - Word-Level):**
1. **data_detector.py**: Scans originals/ directory for new writers to process
2. **version_manager.py**: Creates new dataset version (v1, v2, etc.) with metadata
3. **segmentation_runner.py**: Converts raw scans to 384x384 segmented word images (will support line-level)
4. **Manual Quality Control**: Edit segmented images to remove problematic content
5. **annotation_creator.py**: Extracts ground truth from filenames to annotations.json
6. **augmentation_manager.py**: Applies optional data augmentation for training robustness
7. **dataset_splitter.py**: Creates TrOCR-compatible train/val/test splits to gt_*.txt files
8. **Validation & Cleanup**: Verifies output files and removes old dataset versions

**Dataset Output Structure (Current Word-Level, Will Change to Line-Level):**
```
trocr_ready_data/
├── v1/                          # LEGACY: Word-level dataset
│   ├── images/                    # All 384x384 segmented word images (JPG)
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
├── v2/                          # Synthetic multi-word lines (temporary solution)
│   └── ...
└── v3+/                         # Future: Real line-level data from updated orchestrator
```

**TrOCR Format Example (gt_train.txt):**
```
images/writer01_page01_001_Åsa.jpg	Åsa
images/writer01_page01_002_huvudgång.jpg	huvudgång
images/writer01_page02_025_KAPELL.jpg	KAPELL
```
**Segmentation Features:**
- **Dynamic image analysis**: Automatically detects DPI and adjusts parameters accordingly
- **Reference marker detection**: Uses circular markers for precise coordinate transformation
- **Adaptive parameters**: Automatically scales detection based on actual image resolution (200 DPI vs 300 DPI)
- **Smart margin control**: 6-pixel inward margin to remove border artifacts from segmented words
- **Fallback mode**: Functions with simple coordinate conversion when markers aren't detected
- **Visualization support**: Generate debug images showing detected markers and segmentation regions

**Segmentation Options:**
- `--use-references`: Enable reference marker detection (default: True)
- `--no-references`: Disable markers, use simple coordinate conversion
- `--visualize`: Generate visualization images for debugging
- `--viz-output`: Custom directory for visualization output

## Important Usage Notes

All scripts must be run from the project root directory using Python module syntax:

```bash
# Main orchestrator pipeline (recommended)
python -m scripts.data_processing.orchestrator.main --auto-detect

# Alternative command options:
python -m scripts.data_processing.orchestrator.main --writers writer_01,writer_02 --no-augmentation
python -m scripts.data_processing.orchestrator.main --auto-detect --dry-run --keep-versions 5

# Individual modules (for development/debugging)
python -m scripts.data_processing.template_generator.generate_templates
python -m scripts.data_processing.image_segmentation.segment_images [options]
```

The project uses centralized path configuration in `config/paths.py`. All file paths are relative to the project root, ensuring compatibility with cloud platforms and other deployment environments.

## Model Training

The system includes a complete TrOCR fine-tuning pipeline with cloud platform compatibility, comprehensive metrics evaluation, and intelligent path management.

### Training Pipeline Features
- **HuggingFace Seq2SeqTrainer**: Optimized training loop with automatic mixed-precision and gradient accumulation
- **Custom TrOCR Data Collator**: Handles pixel_values (images) + labels (text tokens) format
- **Swedish-specific metrics**: Character Error Rate (CER), Word Error Rate (WER), BLEU evaluation
- **VisionEncoderDecoderModel**: Based on Microsoft documentation
- **WandB integration**: Experiment tracking and performance monitoring
- **Comprehensive evaluation**: Multi-metric evaluation optimized for Swedish handwriting recognition

### Quick Start Training

```bash
# Dry run with small dataset (10 samples) - test setup
python -m scripts.training.train_model --dry_run --epochs 1

# Standard training with Riksarkivet base model (RECOMMENDED for Swedish)
python -m scripts.training.train_model --epochs 30

# Training with WandB experiment tracking
python -m scripts.training.train_model --epochs 30 --wandb --project_name swedish-handwriting-ocr

# Custom training parameters
python -m scripts.training.train_model --batch_size 8 --epochs 30 --learning_rate 3e-5

# Model/processor combinations (experimental - test different combinations)
python -m scripts.training.train_model --model_combo ra_proc_ra_model  # DEFAULT
python -m scripts.training.train_model --model_combo ra_proc_ms_model --dry_run
python -m scripts.training.train_model --model_combo ms_proc_ra_model --dry_run
python -m scripts.training.train_model --model_combo ms_proc_ms_model --dry_run
```

### Model/Processor Combinations

Use `--model_combo` to select processor and model combination:

- **`ra_proc_ra_model`** (DEFAULT): Riksarkivet processor + Riksarkivet model
  - Best for Swedish historical handwriting
  - Pre-trained on Swedish documents
  - Optimized tokenizer for Swedish characters (å, ä, ö)
  
- `ra_proc_ms_model`: Riksarkivet processor + Microsoft model
  - Swedish tokenizer + modern handwriting model
  - Experimental combination
  
- `ms_proc_ra_model`: Microsoft processor + Riksarkivet model
  - Standard tokenizer + Swedish pre-trained weights
  - May lose Swedish-specific optimizations
  
- `ms_proc_ms_model`: Microsoft processor + Microsoft model
  - Baseline comparison (non-Swedish optimized)

**Recommendation:** Use default `ra_proc_ra_model` for best Swedish performance. Other combinations are available for experimental comparison.

### Training Configuration (Validated)
- **Base model**: Riksarkivet/trocr-base-handwritten-hist-swe-2 (DEFAULT)
- **Architecture**: VisionEncoderDecoderModel
- **Batch size**: 8 (default, configurable)
- **Learning rate**: 3e-5 (default, configurable)
- **Epochs**: 30 (default, configurable)
- **Mixed precision**: FP16 enabled
- **Evaluation strategy**: Per-epoch with CER-based best model selection
- **Data collation**: Custom TrOCRDataCollator for pixel_values + labels format

### Trained Models Available
Trained models are saved with timestamps and version information:
```
models/
└── trocr-swedish-handwriting-v{version}-{timestamp}/
    ├── final_model/              # Complete trained model
    │   ├── model.safetensors     # Model weights
    │   ├── config.json           # Model configuration
    │   ├── tokenizer.json        # Swedish tokenizer
    │   └── preprocessor_config.json
    └── checkpoint-{N}/           # Training checkpoints (optional)
```

### Cloud Platform Deployment
The training pipeline automatically detects cloud environments and adapts paths accordingly:

```bash
# Local development (automatic detection)
python -m scripts.training.train_model --epochs 30

# Cloud platforms (automatic detection)
# Works on RunPod, Google Colab, Azure ML, AWS SageMaker, etc.
# No code changes needed - paths.py handles environment detection
```

**Supported Cloud Platforms:**
- **Azure ML**: Primary cloud platform (enterprise ML platform)
- **RunPod**: GPU instances for cost-effective training
- **Google Colab**: Free and Pro tiers with GPU access
- **AWS SageMaker**: Amazon's ML training service
- **Other cloud platforms**: Automatic detection based on environment

### Training Metrics and Evaluation
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

### Model Output Structure
Trained models are saved with version information and timestamps:
```
models/
└── trocr-swedish-handwriting-v{version}-{timestamp}/
    ├── final_model/          # Complete model and tokenizer
    │   ├── model.safetensors or pytorch_model.bin
    │   ├── config.json
    │   └── tokenizer files
    └── checkpoint-{N}/       # Training checkpoints (optional)
```

The training system uses intelligent path detection from `config/paths.py`:
- **Local development**: Uses dataset versions from `dataset/trocr_ready_data/`
- **Cloud platforms**: Automatically detects cloud environment and adjusts paths
- **Version management**: Always uses latest dataset version automatically

## Model Evaluation

The system includes a comprehensive evaluation framework that automatically adapts to your environment for optimal testing workflow.

### Environment-Aware Evaluation

**Local Development (Single Image Testing):**
```bash
# Basic evaluation (auto-detects latest model)
python -m scripts.evaluation.evaluate_model

# Specify custom model
python -m scripts.evaluation.evaluate_model --model-path models/specific-model

# Force CPU usage
python -m scripts.evaluation.evaluate_model --device cpu
```

**Cloud Platforms (Full Test Split Evaluation):**
```bash
# Same command - automatically detects cloud environment
python -m scripts.evaluation.evaluate_model --output evaluation_results.json

# Note: If Swedish characters (å, ä, ö) display as � in terminal,
# this is a terminal encoding issue, NOT a model problem.
# The model predictions are correct - check the JSON output file
# which preserves UTF-8 encoding properly.
```

### Evaluation Features

**Local Mode (Development):**
- **Random test image**: Selects random image from gt_test.txt (proper test split)
- **Ground truth comparison**: Shows prediction vs actual text with guaranteed matching
- **Quick feedback**: Perfect for development and debugging
- **Simple output**: Clean comparison without overwhelming metrics

**Cloud Mode (Production):**
- **Full test split**: Evaluates entire test dataset from gt_test.txt
- **Comprehensive metrics**: CER, WER, BLEU, Swedish character accuracy, exact match
- **Progress tracking**: Real-time progress updates during evaluation
- **Detailed results**: Per-image predictions with ground truth comparison
- **Export capability**: Save results to JSON for analysis

### Example Outputs

**Local Evaluation Output:**
```
=== LOCAL EVALUATION ===
Image: writer05_page02_036_Björk.jpg
Predicted: 'Bronze'
Ground Truth: 'Björk'
Match: ✗
```

**Cloud Evaluation Metrics:**
```
=== EVALUATION RESULTS ===
CER: 0.0234
WER: 0.1250
BLEU: 0.8567
Swedish chars accuracy: 0.9234
Exact match accuracy: 0.7890
```

### Command Options

```bash
# Basic usage
python -m scripts.evaluation.evaluate_model

# Specify model path
python -m scripts.evaluation.evaluate_model --model-path models/my-model

# Device selection
python -m scripts.evaluation.evaluate_model --device cuda    # Force GPU
python -m scripts.evaluation.evaluate_model --device cpu     # Force CPU
python -m scripts.evaluation.evaluate_model --device auto    # Auto-detect (default)

# Save results to file
python -m scripts.evaluation.evaluate_model --output results.json
```

### Evaluation Metrics

1. **Character Error Rate (CER)**: Percentage of character-level errors
2. **Word Error Rate (WER)**: Percentage of word-level errors
3. **BLEU Score**: Text generation quality (0-1, higher is better)
4. **Swedish Character Accuracy**: Specific accuracy for å, ä, ö characters
5. **Exact Match Accuracy**: Percentage of perfectly predicted texts

### Auto-Detection Features

- **Environment detection**: Automatically adapts behavior for local vs Azure ML
- **Model auto-detection**: Uses latest trained model if none specified
- **Device auto-selection**: Chooses optimal device (GPU if available, CPU fallback)
- **Dataset auto-loading**: Finds test images and ground truth automatically
- **Ground truth loading**: Automatically locates `gt_test.txt` files

### Integration with Training

The evaluation system seamlessly integrates with the training pipeline:
- **Uses same metrics**: Consistent CER, WER, BLEU calculations as training
- **Same data format**: Works with ground truth files from orchestrator pipeline
- **Same path resolution**: Follows identical pattern as dataset_loader for guaranteed compatibility
- **Proper test isolation**: Only evaluates images from gt_test.txt (never train/val data)
- **Version compatibility**: Automatically works with latest dataset versions
- **Model compatibility**: Evaluates any model trained with the training pipeline

## Synthetic Line Generation (v1 → v2)

**TEMPORARY SOLUTION**: Generate synthetic multi-word text lines from existing v1 word images.

> **Note**: This is a temporary bridge solution to enable multi-word training NOW while we collect real line-level data. Future orchestrator updates will process real scanned text lines directly, making this script obsolete.

### Quick Start

```bash
# Test single line first (visual inspection!)
python -m scripts.data_processing.synthetic_data.line_generator \
    --test-single --test-writer writer01 --test-words 5

# Check output: test_line.jpg (should be 1300×256px)
# Verify: 1) Clean cropping, 2) Consistent case, 3) Natural spacing

# If test looks good, run full generation (v1 → v2)
python -m scripts.data_processing.synthetic_data.line_generator
```

### How It Works

Converts v1 word images into v2 synthetic text lines:

1. **Load all word images** from v1 (train+val+test splits)
2. **Crop each word** using threshold-based bounding boxes
3. **Combine words** per line (30-50 characters, typically 4-7 words)
4. **Scale to target width** (1300px, proportional height scaling)
5. **Add vertical padding** to 256px height
6. **Add natural spacing** (8-15px scaled proportionally)
7. **Apply augmentation** (×4: rotation, blur, brightness/contrast)
8. **Split dataset** (70/15/15 train/val/test)
9. **Output to v2** directory

### Key Features

- **Temporary bridge solution**: v1 word-level → v2 line-level (synthetic)
- **Character-based length control**: 30-50 characters per line
- **Production-realistic dimensions**: 1300×256px (matches YOLO line detection)
- **Natural spacing**: Scaled proportionally to maintain visual consistency
- **Same-writer lines**: Each line uses words from one writer only
- **Case grouping**: Separate lines for UPPERCASE, lowercase, and Mixed case
- **Augmentation**: Same pipeline as v1 (×4 multiplier)
- **Expected output**: ~765 base lines → ~3,825 total with augmentation

### Performance Results (v2 Synthetic Dataset)

Training on 3,825 synthetic lines achieved exceptional results on test split:

**Metrics (Test Split - 574 samples):**
- **CER**: 0.19% (Character Error Rate)
- **WER**: 0.52% (Word Error Rate)
- **Exact Match**: 97.21% (perfect predictions)
- **Swedish Character Accuracy**: 99.87% (å, ä, ö recognition)
- **Word Accuracy**: 99.39%

**Most Challenging Words:**
- Compound names with hyphens: ÅSA-LENA, ÅSA-MAJ
- Double consonants: DUETT (often predicted as DUET)
- Context-dependent words: GRAVPLATS, ANDAKT

**Example Predictions:**
```
Prediction: 'program bibel sång avtal gravvård styrka besked'
Prediction: 'VÄSTERÅS TREDJE FÖREVIGT GUD SÖDRA UNDERTECKNAD'
Prediction: 'instrumental västerås trygghet blomsterarrangemang'
Prediction: 'kistbärare andra gemenskap klockringning'
```

**Important Note - Validation Pending:**
⚠️ These results are on synthetic test data (same distribution as training). **Real-world performance** on YOLO-detected lines from actual scanned documents may differ significantly due to:
- Different handwriting variations not seen in training
- YOLO cropping artifacts and boundary effects
- Text outside the limited funeral-domain vocabulary
- Natural line breaks and spacing variations

**Next Steps:**
1. ✅ Baseline established (CER 0.19% on synthetic data)
2. ⏳ Test with YOLO line detection on real documents
3. ⏳ Evaluate real-world CER (expected: 5-15% based on domain shift)
4. ⏳ Collect more diverse line-level data if needed
5. ⏳ Retrain with expanded dataset for production deployment

### Output Structure

```
dataset/trocr_ready_data/
├── v1/                          # Original word images (LEGACY)
│   └── ...
└── v2/                          # Generated synthetic lines (TEMPORARY)
    ├── images/                  # Base line images (1300×256px, ~765 lines)
    ├── images_augmented/        # Augmented versions (~3,060 lines)
    ├── gt_train.txt            # 70% training data
    ├── gt_val.txt              # 15% validation data
    └── gt_test.txt             # 15% test data
```

> **Future**: v3+ will contain real line-level data from updated orchestrator (not synthetic).

### Command Options

```bash
--test-single              Generate single test line (outputs: test_line.jpg)
--test-writer WRITER_ID    Use specific writer for test (default: random)
--test-words NUM           Number of words in test line (default: 5)
--show-image               Display test image (requires X11/display)
```

## System Features

### Data Processing Capabilities
- **Comprehensive Swedish vocabulary**: 150+ categorized words covering Swedish characters (å, ä, ö), names, places, dates, and specialized terminology (LEGACY: word-level, will expand for line-level)
- **Intelligent template generation**: PDF templates with reference markers, page numbers, and scanning instructions (LEGACY: will be updated for line-level collection)
- **Advanced image segmentation** (LEGACY - Word-Level):
  - Automatic DPI detection and parameter adaptation
  - Reference marker detection for precise coordinate mapping
  - Dynamic margin adjustment (6px inward) to remove border artifacts
  - Fallback coordinate transformation when markers aren't available
- **Intelligent data pipeline**:
  - Automated annotation generation from filename metadata
  - Stratified dataset splitting ensuring writer balance and representation
  - Configurable data augmentation (rotation, blur, brightness/contrast)
  - Version-controlled dataset management with incremental updates
- **Synthetic line generation** (TEMPORARY):
  - Multi-word line generation from word-level data (bridge solution)
  - Character-based length control (30-50 chars)
  - Production-realistic dimensions (1300×256px)
  - Natural spacing scaled proportionally

### TrOCR Optimization
- **Standard image preprocessing**: TrOCR processor handles resizing to 384×384 automatically
- **Tab-separated format**: gt_*.txt files following Microsoft TrOCR standard
- **Proper data distribution**: 70/15/15 train/validation/test splits for robust model training
- **Clean filename format**: Consistent naming for reliable parsing
- **Quality assurance**:
  - Visualization tools for debugging marker detection (LEGACY - word-level)
  - Consistent font rendering across all template text (LEGACY - word-level)
  - Optimized line thickness for clean segmentation (LEGACY - word-level)

### Production Ready Features
- **Configurable pipeline**: Suitable for large-scale dataset generation
- **Complete TrOCR pipeline**: Production-ready fine-tuning with cloud platform compatibility
- **Intelligent path management**: Automatic environment detection for local and cloud deployment
- **Advanced metrics evaluation**: Swedish-specific accuracy measurements for å, ä, ö characters
- **Environment awareness**: Seamless transition between development and production environments
- **Version control**: Incremental dataset versioning with automatic cleanup
