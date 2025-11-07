# Swedish Handwriting OCR

A production-ready system for training and deploying TrOCR models for Swedish handwritten line-level text recognition. The system provides comprehensive tools for template generation, data collection, preprocessing, model training, and evaluation with cloud platform compatibility.

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
│   │   └── writer_01/ to writer_13/
│   └── trocr_ready_data/        # Complete TrOCR-ready datasets by version
│       ├── v2/                  # Synthetic line-level data (bridge solution)
│       │   ├── images/          # Line images with page-width dimensions
│       │   ├── gt_train.txt     # Training data (tab-separated)
│       │   ├── gt_val.txt       # Validation data
│       │   └── gt_test.txt      # Test data
│       └── v3+/                 # Real line-level data from scanned templates
├── scripts/
│   ├── data_processing/         # Data processing and orchestration
│   │   ├── orchestrator/        # Main pipeline orchestration
│   │   ├── template_generator/  # Line-level PDF template generation
│   │   ├── image_segmentation/  # Line-level image processing and segmentation
│   │   └── data_preparation/    # Line preprocessing utilities
│   ├── training/                # TrOCR model training pipeline
│   │   ├── train_model.py          # Main training script with cloud support
│   │   ├── dataset_loader.py       # Swedish handwriting dataset loader
│   │   └── evaluation/             # Training metrics
│   └── evaluation/              # Model evaluation and testing scripts
├── models/
│   └── final_model/             # Trained models
├── docs/
│   └── data_collection/         # Data collection templates and resources
│       ├── generated_templates/ # PDF templates with metadata
│       └── line_texts/          # Swedish sentences for line-level training
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
# Generate line-level templates with Swedish sentences
python -m scripts.data_processing.template_generator.generate_templates

# Process scanned handwritten forms
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
- **Template Generation**: Creates handwriting collection forms with line-level layouts and reference markers
- **Image Segmentation**: Automatically processes scanned documents with line-level preprocessing
- **Quality Control**: Interactive and automated tools for dataset curation
- **Data Orchestration**: End-to-end pipeline from templates to TrOCR-ready datasets

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

## Data Collection and Processing

The system uses curated Swedish sentence collections in `docs/data_collection/line_texts/` containing categorized text optimized for handwriting recognition, with focus on Swedish characters (å, ä, ö).

### Template Generation and Data Collection
**Generate handwriting collection templates:**
```bash
python -m scripts.data_processing.template_generator.generate_templates
```

**Template features:**
- Line-level layouts with Swedish sentences
- Reference markers for automatic coordinate transformation
- Support for both single-row and text-field formats
- Optimized for scanning and automated processing

**Collection workflow:**
1. Print generated PDF templates
2. Distribute to writers for completion
3. Scan completed forms as JPG files in correct page order

### Automated Data Processing
**Run the complete processing pipeline:**
```bash
# Comprehensive pipeline
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
- **Line-level segmentation**: Page-width line images for optimal training
- **Automatic annotations**: Extracts ground truth from template metadata
- **Data augmentation**: Optional rotation, blur, brightness/contrast variations
- **Version management**: Incremental dataset versions with cleanup

### Quality Control
Interactive quality control during processing:
```bash
# During orchestrator execution, interactive prompts allow removal of problematic images
# Format: writer01:problematic_text, writer02:another_issue
```

### Latest Evaluation Results
*[To be filled with current model performance metrics]*

- **Character Error Rate (CER)**: [Pending]
- **Word Error Rate (WER)**: [Pending]  
- **BLEU Score**: [Pending]
- **Swedish Character Accuracy**: [Pending]

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

### Quick Start Training

```bash
# Test training setup with small dataset
python -m scripts.training.train_model --dry_run --epochs 1

# Standard training with base model
python -m scripts.training.train_model --epochs 30

# Training with experiment tracking
python -m scripts.training.train_model --epochs 30 --wandb --project_name swedish-handwriting-ocr

# Custom training parameters
python -m scripts.training.train_model --batch_size 8 --epochs 30 --learning_rate 3e-5
```

### Training Configuration
- **Base model**: Riksarkivet/trocr-base-handwritten-hist-swe-2 (optimized for Swedish)
- **Architecture**: VisionEncoderDecoderModel
- **Batch size**: 8 (configurable)
- **Learning rate**: 3e-5 (configurable)
- **Mixed precision**: FP16 enabled
- **Evaluation strategy**: Per-epoch with CER-based best model selection

### Training Metrics
The training pipeline includes comprehensive Swedish handwriting metrics:
1. **Character Error Rate (CER)**: Core metric for OCR evaluation
2. **Word Error Rate (WER)**: Word-level accuracy measurement
3. **BLEU Score**: Text generation quality assessment
4. **Swedish Character Accuracy**: Specialized metric for å, ä, ö recognition
5. **Exact Match Accuracy**: Perfect prediction percentage

## Model Evaluation

The system includes a comprehensive evaluation framework that adapts to your environment.

### Quick Start Evaluation

```bash
# Basic evaluation (auto-detects latest model)
python -m scripts.evaluation.evaluate_model

# Specify custom model
python -m scripts.evaluation.evaluate_model --model-path models/specific-model

# Save results to file
python -m scripts.evaluation.evaluate_model --output evaluation_results.json
```

### Evaluation Features
- **Environment detection**: Automatically adapts behavior for local vs cloud environments
- **Model auto-detection**: Uses latest trained model if none specified
- **Comprehensive metrics**: CER, WER, BLEU, Swedish character accuracy, exact match
- **Test split isolation**: Only evaluates images from gt_test.txt (proper data isolation)
- **Swedish character support**: Correct handling of å, ä, ö in predictions

### Evaluation Metrics
1. **Character Error Rate (CER)**: Percentage of character-level errors
2. **Word Error Rate (WER)**: Percentage of word-level errors
3. **BLEU Score**: Text generation quality (0-1, higher is better)
4. **Swedish Character Accuracy**: Specific accuracy for å, ä, ö characters
5. **Exact Match Accuracy**: Percentage of perfectly predicted texts

## System Features

### Data Processing Capabilities
- **Comprehensive Swedish text collections**: Categorized sentences covering Swedish characters (å, ä, ö), varied vocabulary and domains
- **Intelligent template generation**: PDF templates with reference markers, line-level layouts, and automated processing instructions
- **Advanced line-level segmentation**: Reference marker detection for precise coordinate mapping, dynamic parameter adaptation, fallback transformation modes
- **Intelligent data pipeline**: Automated annotation generation, stratified dataset splitting, configurable augmentation, version-controlled management

### TrOCR Optimization
- **Standard image preprocessing**: TrOCR processor handles automatic resizing
- **Tab-separated format**: gt_*.txt files following Microsoft TrOCR standard
- **Proper data distribution**: 70/15/15 train/validation/test splits for robust model training
- **Clean filename format**: Consistent naming for reliable parsing

### Production Ready Features
- **Configurable pipeline**: Suitable for large-scale dataset generation
- **Complete TrOCR pipeline**: Production-ready fine-tuning with cloud platform compatibility
- **Intelligent path management**: Automatic environment detection for local and cloud deployment
- **Advanced metrics evaluation**: Swedish-specific accuracy measurements for å, ä, ö characters
- **Environment awareness**: Seamless transition between development and production environments
- **Version control**: Incremental dataset versioning with automatic cleanup
