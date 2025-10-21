# Swedish Handwriting OCR

A production-ready system for training and deploying TrOCR models for Swedish handwritten text recognition. The system provides comprehensive tools for data collection, preprocessing, model training, and evaluation with Azure ML compatibility.

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
# Generate training data from scanned forms
python -m scripts.data_processing.orchestrator.main --auto-detect

# Train TrOCR model on Swedish handwriting
python -m scripts.training.train_model --epochs 10 --wandb

# Evaluate trained model performance
python -m scripts.evaluation.evaluate_model
```

## System Components

### Data Processing Pipeline
**Location**: `scripts/data_processing/`

**Core Components**:
- **Template Generation**: Creates handwriting collection forms with reference markers
- **Image Segmentation**: Automatically processes scanned documents into individual word images
- **Quality Control**: Interactive and automated tools for dataset curation
- **Data Orchestration**: End-to-end pipeline from raw scans to TrOCR-ready datasets

### Training Pipeline
**Location**: `scripts/training/`

**Features**:
- **HuggingFace Integration**: Seq2SeqTrainer with automatic mixed-precision training
- **Swedish Optimization**: Custom metrics and tokenization for Swedish characters (å, ä, ö)
- **Azure ML Support**: Automatic environment detection for cloud deployment
- **Model Persistence**: Complete model saving with tokenizer and configuration

### Evaluation System
**Location**: `scripts/evaluation/`

**Capabilities**:
- **Multi-metric Assessment**: CER, WER, BLEU, and Swedish-specific accuracy
- **Environment Awareness**: Adapts between local development and production evaluation
- **Ground Truth Validation**: Comprehensive test split evaluation with proper data isolation
- **Error Analysis**: Detailed performance analysis and failure pattern identification

## Data Collection and Processing

The system uses curated Swedish word collections in `docs/data_collection/word_collections/` containing categorized vocabulary optimized for handwriting recognition, including specialized terminology with focus on Swedish characters (å, ä, ö).

### Phase 1: Template Generation and Data Collection
**Generate handwriting collection templates:**
```bash
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

### Phase 2: Automated Data Processing
**Run the complete processing pipeline:**
```bash
# Comprehensive pipeline (recommended)
python -m scripts.data_processing.orchestrator.main --auto-detect

# Process specific writers
python -m scripts.data_processing.orchestrator.main --writers writer_01,writer_02

# Preview actions without execution
python -m scripts.data_processing.orchestrator.main --auto-detect --dry-run
```

**Pipeline capabilities:**
- **Complete automation**: Auto-detects new writers and processes end-to-end
- **TrOCR-compatible output**: Creates `gt_train.txt`, `gt_val.txt`, `gt_test.txt` files
- **Proper dataset splitting**: 70/15/15 train/val/test distribution
- **Flat output structure**: `trocr_ready_data/vX/images/` optimized for TrOCR training
- **Clean filename format**: `writer01_page01_001_text.jpg` format
- **384x384 preprocessing**: Integrated TrOCR-ready image formatting
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

**Pipeline Processing Steps:**
1. **data_detector.py**: Scans originals/ directory for new writers to process
2. **version_manager.py**: Creates new dataset version (v1, v2, etc.) with metadata
3. **segmentation_runner.py**: Converts raw scans to 384x384 segmented word images
4. **Manual Quality Control**: Edit segmented images to remove problematic content
5. **annotation_creator.py**: Extracts ground truth from filenames to annotations.json
6. **augmentation_manager.py**: Applies optional data augmentation for training robustness
7. **dataset_splitter.py**: Creates TrOCR-compatible train/val/test splits to gt_*.txt files
8. **Validation & Cleanup**: Verifies output files and removes old dataset versions

**Dataset Output Structure:**
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

The project uses centralized path configuration in `config/paths.py`. All file paths are relative to the project root, ensuring compatibility with Azure ML and other deployment environments.

## Model Training

The system includes a complete TrOCR fine-tuning pipeline with Azure ML compatibility, comprehensive metrics evaluation, and intelligent path management.

### Training Pipeline Features
- **HuggingFace Seq2SeqTrainer**: Optimized training loop with automatic mixed-precision and gradient accumulation
- **Custom TrOCR Data Collator**: Handles pixel_values (images) + labels (text tokens) format
- **Swedish-specific metrics**: Character Error Rate (CER), Word Error Rate (WER), BLEU evaluation
- **VisionEncoderDecoderModel**: Based on Microsoft documentation
- **WandB integration**: Experiment tracking and performance monitoring
- **Comprehensive evaluation**: Multi-metric evaluation optimized for Swedish handwriting recognition

### Quick Start Training

```bash
# Dry run with small dataset (10 samples)
python -m scripts.training.train_model --dry_run --epochs 1

# Full training with WandB logging
python -m scripts.training.train_model --epochs 10 --wandb

# Custom training parameters
python -m scripts.training.train_model --batch_size 8 --epochs 5 --learning_rate 1e-5
```

### Training Configuration (Validated)
- **Base model**: microsoft/trocr-base-handwritten
- **Architecture**: VisionEncoderDecoderModel (updated from TrOCRForCausalLM)
- **Batch size**: 16 (with gradient accumulation for effective batch size of 32)
- **Learning rate**: 5e-5 with warmup
- **Mixed precision**: FP16 enabled with use_fast=True for TrOCRProcessor
- **Evaluation strategy**: Every 200 steps with CER-based best model selection
- **Data collation**: Custom TrOCRDataCollator for pixel_values + labels format

### Trained Models Available
Current trained models ready for evaluation:
```
models/
├── trocr-swedish-handwriting-v3-20251020_091130/
│   ├── final_model/              # Complete trained model (1.3GB)
│   │   ├── model.safetensors     # Model weights
│   │   ├── config.json           # Model configuration
│   │   ├── tokenizer.json        # Swedish tokenizer
│   │   └── preprocessor_config.json
│   └── checkpoint-1/             # Training checkpoint
└── (other training runs...)
```

### Azure ML Deployment
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
└── trocr-swedish-handwriting-v3-20241017_143052/
    ├── final_model/          # Complete model and tokenizer
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── tokenizer files
    └── checkpoints/          # Training checkpoints
```

The training system uses intelligent path detection from `config/paths.py`:
- **Local development**: Uses dataset versions from `dataset/trocr_ready_data/`
- **Azure ML**: Automatically detects mounted data and adjusts paths
- **Version management**: Always uses latest dataset version automatically

## Azure ML Integration

This project supports cloud-based training and deployment through Azure Machine Learning for scalable model development.

### Setup Prerequisites

```bash
# Install Azure ML CLI and SDK
pip install azure-ai-ml azure-cli

# Authenticate with Azure
az login

# Configure Azure CLI defaults (saves typing resource group repeatedly)
az config set defaults.group=<your-resource-group>

# Verify workspace connection
az ml workspace show --name <your-workspace-name> --output table
```

### Local to Cloud Migration

**Quick Start:**
1. **Verify Azure connection** - Test workspace access
2. **Create compute instance** - GPU-enabled instance for training
3. **Upload project** - Via Azure ML Studio or Git
4. **Install dependencies** - Run `pip install -r requirements.txt` on compute instance
5. **Test training pipeline** - Validate environment with dry run

### Development Workflow

**Recommended hybrid approach:**
- **Local development** - Quick testing with small datasets
- **Azure ML training** - Full-scale training on complete dataset
- **Model evaluation** - Both local and cloud-based evaluation

```bash
# Local testing (quick iteration)
python -m scripts.training.train_model --dry_run --epochs 1

# Cloud training (production scale)
az ml job create --file azure-training-job.yml --workspace-name <workspace>

# Model evaluation
python -m scripts.evaluation.evaluate_model --model-path ./models/latest/
```

### Key Features

- **Automated dataset management** - Register and version control training data
- **Scalable compute** - GPU clusters for large-scale training
- **Experiment tracking** - Integration with WandB and Azure ML experiments
- **Model versioning** - Automated model registration and deployment
- **Cost optimization** - Start/stop compute instances as needed

### Environment Configuration

Create custom environment for TrOCR training:

```yaml
# environment.yml
name: trocr-swedish-env
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.8
  - pytorch::pytorch>=1.13.0
  - pytorch::torchvision
  - pip
  - pip:
    - transformers>=4.21.0
    - datasets>=2.0.0
    - pillow>=8.3.0
    - wandb
    - azure-ai-ml
```

### Azure ML vs Local Development

| Feature | Local Development | Azure ML |
|---------|------------------|----------|
| **Dataset Access** | Local file system | Azure ML datastores |
| **Training Command** | Direct Python execution | Azure ML job submission |
| **Model Storage** | Local directory | Azure ML model registry |
| **Compute Resources** | Local GPU | Scalable GPU clusters |
| **Experiment Tracking** | Local logs | Azure ML + WandB integration |
| **Cost** | Hardware ownership | Pay-per-use cloud resources |

For detailed setup instructions and specific Azure configurations, see `TrOCR_Setup_Plan.ipynb`.

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

**Azure ML (Full Test Split Evaluation):**
```bash
# Same command - automatically detects Azure environment
python -m scripts.evaluation.evaluate_model --output evaluation_results.json
```

### Evaluation Features

**Local Mode (Development):**
- **Random test image**: Selects random image from gt_test.txt (proper test split)
- **Ground truth comparison**: Shows prediction vs actual text with guaranteed matching
- **Quick feedback**: Perfect for development and debugging
- **Simple output**: Clean comparison without overwhelming metrics

**Azure Mode (Production):**
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

**Azure Evaluation Metrics:**
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

## System Features

### Data Processing Capabilities
- **Comprehensive Swedish vocabulary**: 150+ categorized words covering Swedish characters (å, ä, ö), names, places, dates, and specialized terminology
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

### TrOCR Optimization
- **384x384 preprocessing**: Integrated in segmentation pipeline
- **Tab-separated format**: gt_*.txt files following Microsoft TrOCR standard
- **Proper data distribution**: 70/15/15 train/validation/test splits for robust model training
- **Clean filename format**: writer01 format for reliable parsing
- **Quality assurance**:
  - Visualization tools for debugging marker detection
  - Consistent font rendering across all template text
  - Optimized line thickness (0.5pt) for clean segmentation

### Production Ready Features
- **Configurable pipeline**: Suitable for large-scale dataset generation
- **Complete TrOCR pipeline**: Production-ready fine-tuning with Azure ML compatibility
- **Intelligent path management**: Automatic environment detection for local and cloud deployment
- **Advanced metrics evaluation**: Swedish-specific accuracy measurements for å, ä, ö characters
- **Environment awareness**: Seamless transition between development and production environments
