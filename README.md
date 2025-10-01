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
│   ├── svenska_ord_lista.txt    # Updated word list for data collection
│   ├── raw_scans/               # Original scanned documents
│   ├── segmented/               # Segmented word images
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── annotations/             # Metadata and labels
│   └── templates/               # Templates for data collection
├── scripts/
│   ├── data_processing/         # Data processing scripts
│   ├── training/                # Training scripts
│   └── evaluation/              # Evaluation scripts
├── models/
│   ├── checkpoints/             # Training checkpoints
│   └── final/                   # Final models
├── docs/                        # Documentation
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

See `dataset/svenska_ord_lista.txt` for a comprehensive list of Swedish words, phrases, and special characters to be used for data collection.

### Phase 1: Data Collection
- Create templates based on the word list
- Collect handwriting from at least 10-20 different writers
- Scan or photograph completed forms

### Phase 2: Data Processing
- Segment scanned documents into individual words
- Create annotations and quality control
- Split into training, validation, and test sets

### Phase 3: Model Training
- Fine-tune TrOCR on Swedish handwriting data
- Evaluate performance
- Optimize hyperparameters

## Usage

Upcoming scripts for:
- Segmentation of scanned documents
- Training TrOCR model
- Evaluation and error analysis
- Inference on new images

## Features

- **Comprehensive Swedish vocabulary**: Covers all Swedish characters (å, ä, ö), names, places, dates, and form-specific terminology
- **Automated segmentation**: Scripts to extract individual words from scanned pages
- **TrOCR integration**: Fine-tuning pipeline using Microsoft's TrOCR model
- **Quality control**: Validation and error analysis tools
- **Production ready**: Configurable pipeline suitable for deployment
