from pathlib import Path
import os

def detect_project_environment():
    """
    Detect if we're running locally or on Azure ML
    
    Returns:
        tuple: (environment_type, project_root_path)
        environment_type: 'local' | 'azure_ml' 
        project_root_path: Path to project root
    """

    # Check for Azure ML environment variables
    azure_indicators = [
        'AZUREML_RUN_ID',
        'AZUREML_EXPERIMENT_ID', 
        'AZUREML_RUN_TOKEN',
        'AZUREML_DATAREFERENCE_data'
    ]

    is_azure = any(var in os.environ for var in azure_indicators)

    if is_azure:
        current_file_parent = Path(__file__).parent.parent

        # Check for mounted dataset
        if 'AZUREML_DATAREFERENCE_data' in os.environ:
            azure_data_mount = Path(os.environ['AZUREML_DATAREFERENCE_data'])

            # If mount contains project structure (dataset/)
            if (azure_data_mount / "dataset").exists():
                return 'azure_ml', azure_data_mount
            
            # If mount IS the dataset directory
            elif (azure_data_mount / "trocr_ready_data").exists():
                # Create virtual project root
                return 'azure_ml', azure_data_mount.parent
            
        # Fallback: use working directory on Azure
        return 'azure_ml', current_file_parent
    
    # Local development
    return 'local', Path(__file__).parent.parent

def get_dataset_root():
    """
    Get dataset root directory - Azure ML aware
    
    Returns:
        Path: Dataset root directory
    """
    env_type, project_root = detect_project_environment()

    if env_type == 'azure_ml':
        # Check for direct dataset mount
        if 'AZUREML_DATAREFERENCE_data' in os.environ:
            azure_data = Path(os.environ['AZUREML_DATAREFERENCE_data'])

            # If Azure mount contains dataset folder
            if (azure_data / "dataset").exists():
                return azure_data / "dataset"
            
            # If Azure mount IS the dataset folder (trocr_ready_data exists)
            elif (azure_data / "trocr_ready_data").exists():
                return azure_data
            
            # Fallback
            return azure_data
        
        # No direct mount, use project structure
        return project_root / "dataset"
    
    # Local development
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
    if ENV_TYPE == 'azure_ml':
        print("Azure ML Environment Variables:")
        for key, value in os.environ.items():
            if 'AZUREML' in key:
                print(f"  {key}: {value}")

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

def get_latest_model() -> Path:
    """Get latest trained model directory"""
    if not ModelPaths.ROOT.exists():
        raise FileNotFoundError(f"Models directory not found: {ModelPaths.ROOT}")

    model_dirs = [d for d in ModelPaths.ROOT.iterdir()
                  if d.is_dir() and d.name.startswith("trocr-swedish-handwriting")]

    if not model_dirs:
        raise FileNotFoundError("No trained models found")

    return max(model_dirs, key=lambda x: x.stat().st_mtime)