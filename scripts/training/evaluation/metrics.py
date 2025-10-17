"""
Metrics for TrOcr SwedishHandwriting - CER, WER, BLEU evaluation
"""

import logging
import numpy as np
from datasets import load_metric
import evaluate
import re
from collections import Counter
from difflib import SequenceMatcher
from transformers import TrOCRProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize metrics (global)
try:
    # Try new evaluation API first
    cer_metric = evaluate.load('cer')
    wer_metric = evaluate.load('wer')
    bleu_metric = evaluate.load('bleu')
    logger.info("Using new evaluate API for metrics")
except Exception as e:
    # Fallback to older dataset API
    logger.warning(f"Evaluate API failed: {e}, trying legacy dataset API")
    cer_metric = load_metric('cer')
    wer_metric = load_metric('wer')
    bleu_metric = load_metric('bleu')
    logger.info("Using legacy dataset API metrics")

# Swedish characters for special analysis
SWEDISH_CHARS = {'å', 'ä', 'ö', 'Å', 'Ä', 'Ö'}

def clean_text(text: str) -> str:
    """
    Clean text for consistent evaluation

    Args:
        text (str). Raw text from model/ground truth

    Returns:
        str: Cleaned text for metric calculation
    """

    if not text:
        return ""

    # Clean whitespace and special chars
    text = text.strip()

    # Clear extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Normalize case
    text = text.upper()

    return text

def extract_swedish_chars(text: str) -> list[str]:
    """
    Extract swedish characters from text

    Args: 
        text (str): Text to analyse

    Returns: 
        list[str]: List of swedish chars which is found
    """
    return [char for char in text if char in SWEDISH_CHARS]

def compute_metrics(eval_pred) -> dict[str, float]:
    """
    Main function used by Seq2SeqTrainer for evaluation

    Args:
        eval_pred: Tuple with (predictions, labels) from Seq2SeqTrainer

    Returns:
        dict[str, float]: Dictionary with all metrics
    """
    predictions, labels = eval_pred

    # Decode predictions and labels to text
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')

    # Decode token IDs to text
    decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    # Clean text for consistent evaluation
    cleaned_preds = [clean_text(pred) for pred in decoded_preds]
    cleaned_labels = [clean_text(label) for label in decoded_labels]

    # Calculate core metrics
    metrics = {}

    # 1. Character Error Rate (CER)
    cer_score = cer_metric.compute(predictions=cleaned_preds, references=cleaned_labels)
    metrics['eval_cer'] = cer_score

    # 2. Word Error Rate (WER)
    wer_score = wer_metric.compute(predictions=cleaned_preds, references=cleaned_labels)
    metrics['eval_wer'] = wer_score

    # 3. BLEU score
    bleu_labels = [[label] for label in cleaned_labels]
    bleu_score = bleu_metric.compute(predictions=cleaned_preds, references=bleu_labels)
    metrics['eval_bleu'] = bleu_score['bleu'] if isinstance(bleu_score, dict) else bleu_score

    # 4. Swedish special chars accuracy
    swedish_accuracy = compute_swedish_accuracy(cleaned_preds, cleaned_labels)
    if swedish_accuracy is not None:
        metrics['eval_swedish_chars'] = swedish_accuracy

    # 5. Exact match accuracy
    exact_matches = sum(1 for pred, label in zip(cleaned_preds, cleaned_labels) if pred == label)
    metrics['eval_exact_match'] = exact_matches / len(cleaned_preds)

    # Log for debugging
    logger.info(f"Evaluation metrics: CER={cer_score:.4f}, WER={wer_score:.4f}, BLEU={metrics['eval_bleu']:.4f}")

    return metrics

def compute_swedish_accuracy(predictions: list[str], references: list[str]) -> float:
    """
    Calculate accuracy specifically for swedish chars (å, ä, ö)

    Args:
        predictions (list[str]): The models' preditions
        references (list[str]): Ground truth

    Returns:
        float: Accuracy of swedish chars (0.0 - 1.0)
    """
    if not predictions or not references:
        return 0.0
    
    total_swedish_chars = 0
    correct_swedish_chars = 0

    for pred, ref in zip(predictions, references):
        # Extract swedish chars from both texts
        pred_swedish = extract_swedish_chars(pred)
        ref_swedish = extract_swedish_chars(ref)

        # Count every swedish char
        pred_counts = Counter(pred_swedish)
        ref_counts = Counter(ref_swedish)

        # Calculate amout of swedish chars in reference
        total_swedish_chars += len(ref_swedish)

        # Compare swedish chars position by position
        for char, ref_count in ref_counts.items():
            pred_count = pred_counts.get(char, 0)
            correct_swedish_chars += min(pred_count, ref_count)

    if total_swedish_chars == 0:
        return None
        
    return correct_swedish_chars / total_swedish_chars

def log_sample_predictions(predictions: list[str], references: list[str], max_samples: int = 5):
    """
    Log example of predictions for debugging
    """
    logger.info("Sample predictions vs references:")
    for i, (pred, ref) in enumerate(zip(predictions[:max_samples], references[:max_samples])):
        logger.info(f"  Sample {i+1}: '{pred}' vs '{ref}'")
        if pred != ref:
            pred_swedish = extract_swedish_chars(pred)
            ref_swedish = extract_swedish_chars(ref)
            logger.info(f"   Swedish chars: {pred_swedish} vs {ref_swedish}")