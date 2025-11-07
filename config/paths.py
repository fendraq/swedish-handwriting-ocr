from pathlib import Path
import os

def detect_project_environment():
    """
    Detect if we're running locally or in cloud
    
    Returns:
        tuple: (environment_type, project_root_path)
        environment_type: 'local' | 'cloud' 
        project_root_path: Path to project root
    """
    
    # Check for local development indicators
    local_indicators = [
        # Windows paths
        'C:\\Users\\',
        'C:/Users/',
        # Linux/Mac home paths  
        '/home/',
        '/Users/',
        # WSL paths
        '/mnt/c/Users/',
    ]
    
    # Check current working directory and parents
    current_path = str(Path.cwd())
    
    # Also check for development tools in parent directories
    has_dev_tools = any(
        (Path.cwd() / tool).exists() or 
        any((parent / tool).exists() for parent in Path.cwd().parents)
        for tool in ['.vscode', 'venv', '.git']
    )
    
    is_local = any(indicator in current_path for indicator in local_indicators) or has_dev_tools
    
    if is_local:
        return 'local', Path(__file__).parent.parent
    else:
        return 'cloud', Path(__file__).parent.parent

def get_dataset_root():
    """
    Get dataset root directory - Cloud aware
    
    Returns:
        Path: Dataset root directory
    """
    env_type, project_root = detect_project_environment()
    
    # For both local and cloud, use standard project structure
    return project_root / "dataset"

# Dynamic roots based on environment detection
ENV_TYPE, PROJECT_ROOT = detect_project_environment()
DATASET_ROOT = get_dataset_root()

def debug_paths():
    """Debug function to show current path configuration"""
    print(f"Environment: {ENV_TYPE}")
    print(f"Project Root: {PROJECT_ROOT}")  
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"TrOCR Ready: {DatasetPaths.TROCR_READY_DATA}")
    print("Paths exist check:")
    print(f"  Dataset Root exists: {DATASET_ROOT.exists()}")
    print(f"  TrOCR Ready exists: {DatasetPaths.TROCR_READY_DATA.exists()}")
    print(f"Current working directory: {Path.cwd()}")
    
    if ENV_TYPE == 'cloud':
        print("Cloud environment detected")
        # Show cloud-specific info if any environment variables exist
        cloud_vars = {k: v for k, v in os.environ.items() 
                     if any(indicator in k for indicator in ['RUNPOD', 'COLAB', 'AZUREML', 'AWS'])}
        if cloud_vars:
            print("Cloud environment variables:")
            for key, value in cloud_vars.items():
                print(f"  {key}: {value}")
    else:
        print("Local development environment detected")

# Project root
DOCS_ROOT = PROJECT_ROOT / "docs"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
CONFIG_ROOT = PROJECT_ROOT / "config"
MODELS_ROOT = PROJECT_ROOT / "models"

# Model Paths
class ModelPaths:
    ROOT = MODELS_ROOT

    # Trained models directory
    TRAINED_MODELS = MODELS_ROOT

    # Current/latest model (auto-detection)
    LATEST_MODEL = None

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

def get_single_line_metadata() -> Path:
    """Get path to latest single-line template metadata."""
    template_dir = DocsPaths.GENERATED_TEMPLATES
    sl_files = list(template_dir.glob("swedish_handwriting_sl_*.json"))
    if not sl_files:
        raise FileNotFoundError("No single-line metadata files found")
    
    # Return the most recent single-line metadata file
    latest_sl_file = sorted(sl_files)[-1]
    return latest_sl_file

def get_version_dir(version: str = None) -> Path:
    """Get specific version directory in trocr_ready_data."""
    if version is None:
        from scripts.data_processing.orchestrator.version_manager import get_latest_version_number
        version = get_latest_version_number()
    return DatasetPaths.TROCR_READY_DATA / version

def get_next_version() -> str:
    """
    Auto-detect latest version and return next version number.
    DEPRECATED: Use version suffixes instead (e.g., v1_lines, v1_extended).
    
    Returns:
        str: Next version (e.g., 'v2' if 'v1' exists, 'v1' if none exists)
    """
    if not DatasetPaths.TROCR_READY_DATA.exists():
        return 'v1'
    
    versions = [d.name for d in DatasetPaths.TROCR_READY_DATA.iterdir() 
                if d.is_dir() and d.name.startswith('v')]
    
    if not versions:
        return 'v1'
    
    # Extract version numbers (v1 -> 1, v2_lines -> 2)
    version_numbers = []
    for v in versions:
        try:
            # Take first part after 'v' and before '_' (if exists)
            num_str = v[1:].split('_')[0]
            version_numbers.append(int(num_str))
        except (ValueError, IndexError):
            continue
    
    if not version_numbers:
        return 'v1'
    
    latest_num = max(version_numbers)
    return f'v{latest_num + 1}'

def get_latest_version() -> Path:
    """Get current/latest version directory."""
    return DatasetPaths.CURRENT_VERSION

def get_version_images(version: str) -> Path:
    """Get images directory for specific version."""
    return DatasetPaths.TROCR_READY_DATA / version / "images"

def get_latest_model() -> Path:
    """Get latest trained model directory"""
    if not ModelPaths.ROOT.exists():
        raise FileNotFoundError(f"Models directory not found: {ModelPaths.ROOT}")

    model_dirs = [d for d in ModelPaths.ROOT.iterdir()
                  if d.is_dir() and d.name.startswith("trocr-swedish-handwriting")]

    if not model_dirs:
        raise FileNotFoundError("No trained models found")

    return max(model_dirs, key=lambda x: x.stat().st_mtime)