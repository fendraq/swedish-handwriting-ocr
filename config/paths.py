from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DOCS_ROOT = PROJECT_ROOT / "docs"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
CONFIG_ROOT = PROJECT_ROOT / "config"

# Dataset paths
class DatasetPaths:
    ROOT = DATASET_ROOT
    
    # Original data
    ORIGINALS = DATASET_ROOT / "originals"
    ORIGINALS_ANNOTATIONS = ORIGINALS / "annotations"
    
    # Processing stages
    SEGMENTED_WORDS = DATASET_ROOT / "segmented_words" # Legacy - to be removed
    PREPROCESSED = DATASET_ROOT / "preprocessed" # Legacy - to be removed
    
    # Splits
    SPLITS = DATASET_ROOT / "splits"
    TRAIN = SPLITS / "train"
    VAL = SPLITS / "val"
    TEST = SPLITS / "test"
    
    # Processed data
    TROCR_READY_DATA = DATASET_ROOT / "trocr_ready_data"
    CURRENT_VERSION = TROCR_READY_DATA / "current"
    LOGS = PROJECT_ROOT / "logs"  # Pipeline logs

    # Azure ready
    AZURE_READY = DATASET_ROOT / "azure_ready" # Legacy - to be removed

# Documentation paths 
class DocsPaths:
    ROOT = DOCS_ROOT
    DATA_COLLECTION = DOCS_ROOT / "data_collection"
    GENERATED_TEMPLATES = DATA_COLLECTION / "generated_templates"
    WORD_COLLECTIONS = DATA_COLLECTION / "word_collections"

# Scripts paths
class ScriptsPaths:
    ROOT = SCRIPTS_ROOT
    DATA_PROCESSING = SCRIPTS_ROOT / "data_processing"
    DATA_PREPARATION = DATA_PROCESSING / "data_preparation"

# Config paths
class ConfigPaths:
    ROOT = CONFIG_ROOT
    CONFIG_YAML = CONFIG_ROOT / "config.yaml"

# Helper functions
def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if not."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_writer_dir(writer_id: str) -> Path:
    """Get writer directory in originals."""
    return DatasetPaths.ORIGINALS / writer_id

def get_writer_segmented_dir(writer_id: str) -> Path:
    """Get writer directory in segmented_words."""
    return DatasetPaths.SEGMENTED_WORDS / writer_id

# Template metadata path (din nuvarande fil)
def get_template_metadata() -> Path:
    """Get path to complete template metadata."""
    return DocsPaths.GENERATED_TEMPLATES / "complete_template_metadata.json"

def get_version_dir(version: str) -> Path:
    """Get specific version directory in trocr_ready_data."""
    return DatasetPaths.TROCR_READY_DATA / version

def get_latest_version() -> Path:
    """Get current/latest version directory."""
    return DatasetPaths.CURRENT_VERSION

def get_version_images(version: str) -> Path:
    """Get images directory for specific version."""
    return DatasetPaths.TROCR_READY_DATA / version / "images"