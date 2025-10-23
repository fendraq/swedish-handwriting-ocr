"""
TrOCR Model Evaluation Script
Auto-detects environment and adapts evaluation mode accordingly
"""
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import argparse
from pathlib import Path
import json
import logging
from datetime import datetime
from config.paths import detect_project_environment, get_latest_model, PROJECT_ROOT, DatasetPaths
from scripts.data_processing.orchestrator.version_manager import get_latest_version_number
from scripts.training.evaluation.metrics import (
    cer_metric, wer_metric, bleu_metric,
    clean_text, compute_swedish_accuracy, log_sample_predictions
)
import random

class TrOCRModelEvaluator:
    """
    Evaluation system for TrOCR Swedish handwriting models.
    Supports both local single-image testing and full dataset evaluation.
    """
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize the TrOCR model evaluator with environment-aware setup.

        Args: 
            model_path: Path to trained model(optional - auto-detects latest if None)
            device: Device to run interence on ('auto', 'cuda', 'cpu')
        """
        # Detect environment
        self.env_type, self.project_root = detect_project_environment()

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        self.device = self._setup_device(device)

        # Auto-detect model path if not provided
        if model_path is None:
            self.model_path = self._find_latest_model()
        else:
            # Handle both absolute and relative paths
            if Path(model_path).is_absolute():
                self.model_path = Path(model_path)
            else:
                self.model_path = PROJECT_ROOT / model_path

        # Load model and processor
        self.model, self.processor = self._load_model_and_processor()

        self.logger.info(f"Environment: {self.env_type}")
        self.logger.info(f"TrOCR Evaluator initialized with model from {self.model_path}")
        self.logger.info(f"Running device: {self.device}")

        # Set evaluation mode based on environment
        """ if self.env_type == 'local':
            self.logger.info("Local environment detected - single image evaluation mode")
        else: """
        self.logger.info("Cloud environment detected - full test split evaluation mode")

    def _setup_device(self, device: str) -> str:
        """
        Setup and validate device for inference

        Args:
            device: Device preference ('auto', 'cuda', 'cpu')

        Returns: 
            str: Validated device string
        """
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                self.logger.info("CUDA not available, using CPU")

        elif device == 'cuda':
            if not torch.cuda.is_available():
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                device = 'cpu'

        elif device == 'cpu':
            self.logger.info("CPU device explicitly requested")

        else: 
            raise ValueError(f"Invalid device: {device}. Use 'auto', 'cuda', or 'cpu'")

        return device
    
    def _find_latest_model(self) -> Path:
        """
        Auto-detect the latest trained model directory.

        Returns: 
            Path: Path to latest model directory
        """
        try:
            latest_model_dir = get_latest_model()

            final_model_path = latest_model_dir / 'final_model'
            self.logger.info(f"Auto-detected latest model: {latest_model_dir.name}/final_model")
            return final_model_path
        
        except FileNotFoundError as e:
            self.logger.error(f"Failed to find model: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error finding model: {e}")
            raise

    def _load_model_and_processor(self):
        """
        Load TrOCR model and processor from model path.
        Handles both pytorch_model.bin and safetensors formats.

        Returns:
            tuple: (model, processor)
        """
        try:
            self.logger.info(f"Loading model from {self.model_path}")

            # Check what model formats are available
            pytorch_model = self.model_path / "pytorch_model.bin"
            safetensors_model = self.model_path / "model.safetensors"
            
            if pytorch_model.exists():
                self.logger.info("Found pytorch_model.bin - using for broad compatibility")
                model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_path, 
                    local_files_only=True
                )
            elif safetensors_model.exists():
                self.logger.info("Found model.safetensors - using modern format")
                model = VisionEncoderDecoderModel.from_pretrained(
                    self.model_path,
                    local_files_only=True
                )
            else:
                raise FileNotFoundError(f"No model file found in {self.model_path}")
                
            model.to(self.device)
            model.eval()

            # Load processor with use_fast=True (following training setup)
            processor = TrOCRProcessor.from_pretrained(self.model_path, use_fast=True)

            self.logger.info(f"Model loaded successfully on {self.device}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            return model, processor
        
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _load_test_data(self) -> list[tuple[Path, str]]:
        """
        Load test data from gt_test.txt (same pattern as dataset_loader)

        Returns:
            list[tuple[Path, str]]: List of (image_path, ground_truth) pairs
        """
        latest_version = get_latest_version_number()
        image_base_path = DatasetPaths.TROCR_READY_DATA / latest_version
        gt_file = image_base_path / 'gt_test.txt'
        
        data = []
        with open(gt_file, 'r', encoding='utf-8') as f:
            for line in f:
                img_path_str, text = line.strip().split('\t', 1)
                img_path = image_base_path / img_path_str  # Absolut path
                data.append((img_path, text))
        
        return data            

    def predict_single_image(self, image_path: str) -> str:
        """
        Predict OCR text for a single image.

        Args: 
            image_path: Path to image file

        Returns: 
            str: predicted text
        """
        try:
            # Load and validate image
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            self.logger.info(f"Processing image: {image_path.name}")

            # Load image
            image = Image.open(image_path).convert('RGB')

            # Process image
            pixel_values = self.processor(image, return_tensors='pt').pixel_values
            pixel_values= pixel_values.to(self.device)

            # Generate prediction
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
                predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            self.logger.info(f"Prediction: '{predicted_text}'")
            return predicted_text

        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            raise

    def get_evaluation_data(self):
        """
        Get evaluation data from gt_test.txt
        Local: Random single test entry
        Cloud: All test data entries

        Returns: 
            tuple or list: (Path, str) tuple (local) or list of (Path, str) tuples (Cloud)
        """
        test_data = self._load_test_data()

        """ if self.env_type == 'local':
            # Random entry from test split
            selected_entry = random.choice(test_data)
            image_path, gt_text = selected_entry
            self.logger.info(f"Selected random test image: {image_path.name}")
            return selected_entry
        else: """
            # All test data when Cloud
        self.logger.info(f"Found {len(test_data)} test entries for evaluation")
        return test_data

    def evaluate_test_split(self) -> dict:
        """
        Evaluate model on full test split

        Returns:
            dict: Comprehensive evaluation results with CER, WER, BLEU, etc.
        """        
        try:
            self.logger.info("=== FULL EVALUATION MODE ===")
            
            # Get all test images
            test_data = self.get_evaluation_data()

            self.logger.info(f"Starting evaluation on {len(test_data)} test entries")

            # Prepare for metrics calculation
            predictions_list = []
            references_list = []
            detailed_results = []
            
            start_time = datetime.now()

            for i, (image_path, gt_text) in enumerate(test_data):
                try:                    
                    # Get corresponding ground truth
                    predicted_text = self.predict_single_image(str(image_path))
                    
                    # Clean texts for metrics
                    clean_pred = clean_text(predicted_text)
                    clean_gt = clean_text(gt_text)
                    
                    predictions_list.append(clean_pred)
                    references_list.append(clean_gt)
                    
                    detailed_results.append({
                        'image_path': str(image_path),
                        'image_name': image_path.name,
                        'predicted_text': predicted_text,
                        'ground_truth': gt_text,
                        'clean_prediction': clean_pred,
                        'clean_ground_truth': clean_gt
                    })

                    # Progress logging
                    if (i + 1) % max(1, len(test_data) // 10) == 0:
                        progress = (i + 1) / len(test_data) * 100
                        self.logger.info(f"Progress: {progress:.1f}% ({i + 1}/{len(test_data)})")

                except Exception as e:
                    self.logger.error(f"Failed to process {image_path}: {e}")
                    continue
            
            # Calculate metrics using metrics.py functions
            if predictions_list and references_list:
                # Core metrics
                cer_score = cer_metric.compute(predictions=predictions_list, references=references_list)
                wer_score = wer_metric.compute(predictions=predictions_list, references=references_list)
                
                # BLEU expects list of lists for references
                bleu_references = [[ref] for ref in references_list]
                bleu_score = bleu_metric.compute(predictions=predictions_list, references=bleu_references)
                bleu_value = bleu_score['bleu'] if isinstance(bleu_score, dict) else bleu_score
                
                # Swedish character accuracy
                swedish_acc = compute_swedish_accuracy(predictions_list, references_list)
                
                # Exact match accuracy
                exact_matches = sum(1 for pred, ref in zip(predictions_list, references_list) if pred == ref)
                exact_match_acc = exact_matches / len(predictions_list)
                
            else:
                cer_score = wer_score = bleu_value = swedish_acc = exact_match_acc = 0.0

            # Final results
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            results = {
                'evaluation_summary': {
                    'total_images': len(test_data),
                    'successful_evaluations': len(predictions_list),
                    'processing_time': processing_time,
                    'avg_time_per_image': processing_time / len(test_data) if test_data else 0,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat()
                },
                'metrics': {
                    'cer': cer_score,
                    'wer': wer_score, 
                    'bleu': bleu_value,
                    'swedish_character_accuracy': swedish_acc,
                    'exact_match_accuracy': exact_match_acc
                },
                'detailed_predictions': detailed_results
            }
            
            # Log metrics
            self.logger.info("=== EVALUATION RESULTS ===")
            self.logger.info(f"CER: {cer_score:.4f}")
            self.logger.info(f"WER: {wer_score:.4f}")
            self.logger.info(f"BLEU: {bleu_value:.4f}")
            if swedish_acc is not None:
                self.logger.info(f"Swedish chars accuracy: {swedish_acc:.4f}")
            self.logger.info(f"Exact match accuracy: {exact_match_acc:.4f}")
            
            # Log sample predictions for debugging
            log_sample_predictions(predictions_list, references_list, max_samples=5)
            
            return results

        except Exception as e:
            self.logger.error(f"Failed to evaluate test split: {e}")
            raise

    def _evaluate_local(self) -> dict:
        """
        Simple local evaluation - just prediction vs ground truth.
        No metrics calculated since single image metrics aren't meaningful.
        
        Returns:
            dict: Simple comparison results
        """
        try:
            # Get random test entry (both image path and ground truth)
            image_path, gt_text = self.get_evaluation_data()
            
            # Get prediction
            predicted_text = self.predict_single_image(str(image_path))
            
            # Simple comparison
            results = {
                'mode': 'local_evaluation',
                'image_name': image_path.name,  
                'image_path': str(image_path),
                'predicted_text': predicted_text,
                'ground_truth': gt_text,
                'match': predicted_text.strip().upper() == gt_text.strip().upper()
            }
            
            # Simple logging output
            self.logger.info("=== LOCAL EVALUATION ===")
            self.logger.info(f"Image: {image_path.name}")
            self.logger.info(f"Predicted: '{predicted_text}'")
            self.logger.info(f"Ground Truth: '{gt_text}'")
            self.logger.info(f"Match: {'✓' if results['match'] else '✗'}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed local evaluation: {e}")
            raise

    def evaluate(self) -> dict:
        """
        Main evaluation method - automatically adapts to environment.
        Local: Single random image with ground truth comparison
        Cloud: Full test split with comprehensive metrics
        
        Returns:
            dict: Evaluation results adapted to environment
        """
        """ if self.env_type == 'local':
            return self._evaluate_local()
        else: """
        return self.evaluate_test_split()
        

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Argument parser
    parser = argparse.ArgumentParser(description="TrOCR Swedish Handwriting Model Evaluation")
    parser.add_argument('--model-path', type=str, help='Path to model (auto-detects latest if not provided)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], 
                       help='Device to run on')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        # Create evaluator and run
        evaluator = TrOCRModelEvaluator(model_path=args.model_path, device=args.device)
        results = evaluator.evaluate()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.output}")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        exit(1)