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
    cer_metric, wer_metric,
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

            processor = TrOCRProcessor.from_pretrained(self.model_path, use_fast=False)
            
            # CRITICAL FIX: Apply same generation config as training
            # This is what makes Swedish subwords work!
            self.logger.info("=== APPLYING GENERATION CONFIG ===")
            model.config.decoder_start_token_id = 0  # Critical for Swedish!
            model.config.bos_token_id = 0
            model.config.eos_token_id = 2
            model.config.pad_token_id = 1
            
            model.generation_config.bos_token_id = 0
            model.generation_config.decoder_start_token_id = 0
            model.generation_config.eos_token_id = 2
            model.generation_config.pad_token_id = 1
            model.generation_config.max_length = 128
            model.generation_config.early_stopping = False
            model.generation_config.num_beams = 4
            model.generation_config.length_penalty = 0.8
            
            self.logger.info(f"Set decoder_start_token_id = {model.config.decoder_start_token_id}")
            self.logger.info(f"Set max_length = {model.generation_config.max_length}")

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
            dict: Comprehensive evaluation results with CER, WER, etc.
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
                
                # Swedish character accuracy
                swedish_acc = compute_swedish_accuracy(predictions_list, references_list)
                
                # Exact match accuracy
                exact_matches = sum(1 for pred, ref in zip(predictions_list, references_list) if pred == ref)
                exact_match_acc = exact_matches / len(predictions_list)

                # Word-level error analysis
                word_analysis = self.compute_word_level_errors(predictions_list, references_list) if predictions_list else {}
                
                # Swedish character confusion matrix
                swedish_confusion = self.compute_swedish_char_confusion(predictions_list, references_list)

                # Generate training recommendations
                training_recommendations = self.generate_training_recommendations(word_analysis, swedish_confusion)
            else:
                cer_score = wer_score = swedish_acc = exact_match_acc = 0.0
                word_analysis = {}
                swedish_confusion = {}
                training_recommendations = {}

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
                    'swedish_character_accuracy': swedish_acc,
                    'exact_match_accuracy': exact_match_acc
                },
                'word_level_analysis': word_analysis,
                'swedish_char_confusion': swedish_confusion,
                'training_recommendations': training_recommendations,
                'detailed_predictions': detailed_results
            }
            
            # Log metrics
            self.logger.info("=== EVALUATION RESULTS ===")
            self.logger.info(f"CER: {cer_score:.4f}")
            self.logger.info(f"WER: {wer_score:.4f}")
            if swedish_acc is not None:
                self.logger.info(f"Swedish chars accuracy: {swedish_acc:.4f}")
            self.logger.info(f"Exact match accuracy: {exact_match_acc:.4f}")

            # Log analysis results
            if word_analysis.get('most_problematic_words'):
                self.logger.info("=== WORD-LEVEL ANALYSIS ===")
                self.logger.info(f"Word accuracy: {word_analysis.get('word_accuracy', 0):.4f}")
                top_problematic = list(word_analysis['most_problematic_words'].items())[:5]
                self.logger.info(f"Most problematic words: {[f'{word}({count})' for word, count in top_problematic]}")

            if training_recommendations.get('high_priority_words'):
                self.logger.info("=== TRAINING RECOMMENDATIONS ===")
                for rec in training_recommendations['high_priority_words'][:3]:
                    self.logger.info(f"  • {rec['word']}: {rec['error_count']} errors - {rec['suggestion']}")

            if training_recommendations.get('swedish_char_issues'):
                self.logger.info("=== SWEDISH CHARACTER ISSUES ===")
                for issue in training_recommendations['swedish_char_issues'][:3]:
                    self.logger.info(f"  • {issue['character']}: {issue['accuracy']:.1%} accuracy")
            
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
        
    def compute_word_level_errors(self, predictions: list[str], references: list[str]) -> dict:
        """
        Analyse errors on word-level to identify problematic words

        Args:
            predictions: List with model predictions
            references: List with correct words

        Returns:
            dict: Detailed word-level analysis
        """
        word_errors = []
        total_words = 0
        correct_words = 0
        word_error_counts = {}

        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()

            total_words += len(ref_words)

            # Compare word position by position (simpified alignment)
            max_len = max(len(pred_words), len(ref_words))

            for i in range(max_len):
                ref_word = ref_words[i] if i < len(ref_words) else None
                pred_word = pred_words[i] if i < len(pred_words) else None

                if ref_word is not None:
                    if pred_word == ref_word:
                        correct_words += 1
                    else:
                        # Count error for this word
                        word_error_counts[ref_word] = word_error_counts.get(ref_word, 0) + 1

                        word_errors.append({
                            'reference': ref_word,
                            'predicted': pred_word if pred_word else "[MISSING]",
                            'error_type': 'substitution' if pred_word else 'deletion'
                        })
                elif pred_word is not None:
                    # Extra word in prediction
                    word_errors.append({
                        'reference': "[NONE]",
                        'predicted': pred_word,
                        'error_type': 'insertion'
                    })
        
        # Sort most problematic word
        most_problematic = dict(sorted(word_error_counts.items(), key=lambda x: x[1], reverse=True)[:20])

        return {
            'word_accuracy': correct_words / total_words if total_words > 0 else 0,
            'total_words': total_words,
            'correct_words': correct_words,
            'total_errors': len(word_errors),
            'error_details': word_errors,
            'most_problematic_words': most_problematic
        }
    
    def compute_swedish_char_confusion(self, predictions: list[str], references: list[str]) -> dict:
        """
        Detailed confusion matrix for swedish chars (å, ä, ö, Å, Ä, Ö)
        
        Args: 
            predictions: List with model predictions
            references: List with correct words

        Returns:
            dict: Confusion matrix for every swedish char
        """
        swedish_chars = ['ä', 'å', 'ö', 'Ä', 'Å', 'Ö']
        confusion_matrix = {}

        for char in swedish_chars:
            confusion_matrix[char] = {}

        for pred, ref in zip(predictions, references):
            # Simple character-by-character comparison
            min_len = min(len(pred), len(ref))

            for i in range(min_len):
                ref_char = ref[i]
                pred_char = pred[i]

                # If reference contains swedish char
                if ref_char in swedish_chars:
                    if pred_char not in confusion_matrix[ref_char]:
                        confusion_matrix[ref_char][pred_char] = 0
                    confusion_matrix[ref_char][pred_char] += 1

        return confusion_matrix
    
    def generate_training_recommendations(self, word_analysis: dict, swedish_confusion: dict) -> dict:
        """
        Generates recommendations for further training based on error analysis

        Args: 
            word_analysis: Results from compute_word_level_errors
            swedish_confusion: Results from compute_swedish_char_confusion

        Returns:
            dict: Structured recommendations for further training
        """
        recommendations = {
            'high_priority_words': [],
            'swedish_char_issues': [],
            'data_expansion_suggestions': []
        }

        # Analyse problematic words
        problematic_words = word_analysis.get('most_problematic_words', {})

        for word, error_count in problematic_words.items():
            if error_count >= 3:
                recommendations['high_priority_words'].append({
                    'word': word,
                    'error_count': error_count,
                    'suggestion': f"Needs {error_count * 2} more examples of '{word}'"
                })

        # Analyse swedish char-problems
        for char, confusions in swedish_confusion.items():
            if confusions:
                total_instances = sum(confusions.values())
                correct_instances = confusions.get(char, 0)

                if total_instances > 0:
                    accuracy = correct_instances / total_instances

                    if accuracy < 0.8 and total_instances >=5:
                        # Find most common error
                        incorrect_chars = {k: v for k, v in confusions.items() if k != char}
                        if incorrect_chars:
                            most_common_error = max(incorrect_chars.keys(), key=lambda k: incorrect_chars[k])

                            recommendations['swedish_char_issues'].append({
                                'character': char,
                                'accuracy': round(accuracy, 3),
                                'total_instances': total_instances,
                                'most_confused_with': most_common_error,
                                'suggestion': f"'{char}' förväxlas ofta med '{most_common_error}' - behöver mer distinkt träningsdata"
                            })

        # Suggestions
        total_word_errors = len(recommendations['high_priority_words'])
        total_char_issues = len(recommendations['swedish_char_issues'])

        if total_word_errors > 10:
            recommendations['data_expansion_suggestions'].append(
                f"A lot of words ({total_word_errors}) have a high error rate - consider a double dataset size"
            )
        if total_char_issues > 3:
            recommendations['data_expansion_suggestions'].append(
                f"Svenska tecken ({total_char_issues}) har problem - fokusera på mer handskriven svenska text"
            )
        
        if total_word_errors == 0 and total_char_issues == 0:
            recommendations['data_expansion_suggestions'].append(
                "Modellen presterar bra - kan utöka med mer varierad text för generalisering"
            )
        
        return recommendations
    
if __name__ == "__main__":
    # Setup logging with UTF-8 encoding for Swedish characters
    import sys
    import io
    
    # Force UTF-8 for stdout FIRST
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    
    # Now configure logging with UTF-8 aware handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler],
        force=True  # Override any existing handlers
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