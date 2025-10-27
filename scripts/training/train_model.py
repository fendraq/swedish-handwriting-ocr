import os
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import logging
import argparse
from pathlib import Path
import wandb
from .dataset_loader import SwedishHandwritingDataset
from config.paths import DatasetPaths
from scripts.data_processing.orchestrator.version_manager import get_latest_version_number
from datetime import datetime
from .evaluation.metrics import create_compute_metrics
from .data_collator import TrOCRDataCollator

class TrainingConfig:
    # Model settings
    model_name = 'microsoft/trocr-base-handwritten'

    # Training settings - Cloud optimized
    batch_size = 8 # from 16 for more stable gradients
    eval_batch_size = 16 # from 32 for consistency
    gradient_accumulation_steps = 4  # from 2 for bigger effective batch
    learning_rate = 3e-5 # from 5e-5 for more constistent learning
    num_epochs = 30 # from 10 for full convergens
    warmup_steps = 500
    
    # Cloud GPU optimizations
    fp16 = True
    dataloader_num_workers = 4
    
    # Evaluation settings
    eval_steps = 1000 # 200
    save_steps = 1000 # from 3000 for more checkpoints
    save_total_limit = 3 # Added for less hdd
    logging_steps = 250 # from 100 for a more balanced logging

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("TrainingConfig initialized")

    # Paths
    @property
    def data_dir(self):
        latest_version = get_latest_version_number()
        if latest_version is None:
            raise FileNotFoundError("No version directories found")
        return DatasetPaths.TROCR_READY_DATA / latest_version
    
    @property
    def output_dir(self):
        latest_version = get_latest_version_number() or "v0"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return DatasetPaths.ROOT.parent / 'models' / f"trocr-swedish-handwriting-{latest_version}-{timestamp}"

def parse_args():
    parser = argparse.ArgumentParser(description="Train TrOCR on Swedish handwriting")
    
    # Training parameters
    parser.add_argument("--dry_run", action="store_true", 
                       help="Run with only 10 samples for testing")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                       help="Learning rate")
    
    # Logging
    parser.add_argument("--wandb", action="store_true",
                       help="Enable WandB experiment tracking")
    parser.add_argument("--project_name", type=str, default="swedish-handwriting-ocr",
                       help="WandB project name")
    
    return parser.parse_args()

def add_swedish_tokens(model, tokenizer, logger):
    """Test if Riksarkivet token configuration fixes Swedish characters"""
    swedish_chars = ["å", "ä", "ö", "Å", "Ä", "Ö"]
    
    # First test: Check current state with Riksarkivet config applied
    logger.info("=== TESTING SWEDISH CHARACTERS WITH RIKSARKIVET CONFIG ===")
    vocab = tokenizer.get_vocab()
    
    for char in swedish_chars:
        if char in vocab:
            token_id = vocab[char]
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
            if decoded == char:
                logger.info(f"'{char}' correctly maps to '{decoded}' (token_id: {token_id})")
            else:
                logger.info(f"'{char}' maps to '{decoded}' (token_id: {token_id})")
        else:
            logger.info(f"'{char}' missing from vocabulary")
    
    # Test full Swedish words
    logger.info("=== TESTING FULL SWEDISH WORDS ===")
    test_words = ["åsa", "kött", "björn", "hälsa", "kärlek", "Göteborg"]
    
    for word in test_words:
        encoded = tokenizer.encode(word, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        if word == decoded:
            logger.info(f"✅ '{word}' -> {encoded} -> '{decoded}' (PERFECT)")
        else:
            logger.info(f"❌ '{word}' -> {encoded} -> '{decoded}' (CORRUPTED)")
    
    # Check if we still need to add tokens
    chars_to_add = []
    for char in swedish_chars:
        if char in vocab:
            token_id = vocab[char]
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
            if decoded != char:
                chars_to_add.append(char)
        else:
            chars_to_add.append(char)
    
    if chars_to_add:
        logger.info(f"Still need to add tokens for: {chars_to_add}")
        logger.info("Proceeding with token addition (legacy approach)...")
        
        # Original token addition logic (kept as fallback)
        old_size = model.decoder.get_input_embeddings().weight.size(0)
        logger.info(f"Decoder embeddings size before: {old_size}")
        
        num_added = tokenizer.add_tokens(chars_to_add)
        logger.info(f"Tokenizer added {num_added} tokens")
        
        model.decoder.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = model.config.decoder.vocab_size
        logger.info(f"Updated model.config.vocab_size to {model.config.vocab_size}")
        
        new_size = model.decoder.get_input_embeddings().weight.size(0)
        logger.info(f"Decoder embeddings size after: {new_size}")
        
        return True
    else:
        logger.info("RIKSARKIVET CONFIG SOLVED THE PROBLEM! No token addition needed!")
        return False
    
def train_model(args):
    """ Main training function """

    # Initialize config
    config = TrainingConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.learning_rate

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.logger.info(f"Using device: {device}")

    # Load model and processor
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    processor = TrOCRProcessor.from_pretrained(config.model_name, use_fast=True)
    
    # Apply Riksarkivet's exact token configuration BEFORE anything else
    config.logger.info("=== APPLYING RIKSARKIVET TOKEN CONFIGURATION ===")
    
    # Configure model config tokens (exact values from Riksarkivet)
    model.config.decoder_start_token_id = 0  # Critical: This triggers the alignment!
    model.config.bos_token_id = 0
    model.config.eos_token_id = 2  
    model.config.pad_token_id = 1
    
    # Configure generation config tokens
    model.generation_config.bos_token_id = 0
    model.generation_config.decoder_start_token_id = 0
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 1
    
    config.logger.info(f"Set model.config.decoder_start_token_id = {model.config.decoder_start_token_id}")
    config.logger.info(f"Set model.config.bos_token_id = {model.config.bos_token_id}")
    config.logger.info(f"Set generation_config.bos_token_id = {model.generation_config.bos_token_id}")
    
    # Add missing Swedish tokens BEFORE validation
    add_swedish_tokens(model, processor.tokenizer, config.logger)
    
    # Configure generation settings (keep existing optimizations)
    model.generation_config.max_length = 128 # from 64, for connected words
    model.generation_config.early_stopping = False # from True to let model generate whole word
    model.generation_config.num_beams = 8 # from 4 for better search
    model.generation_config.length_penalty = 0.8 # New for avoid too short outputs
    
    model.to(device)

    # Create datasets 
    train_dataset = SwedishHandwritingDataset(
        gt_file_path=str(config.data_dir / 'gt_train.txt'),
        processor=processor,
        image_base_path=str(config.data_dir),
        dry_run=args.dry_run
    )

    val_dataset = SwedishHandwritingDataset(
        gt_file_path=str(config.data_dir / "gt_val.txt"),
        processor=processor,
        image_base_path=str(config.data_dir),
        dry_run=args.dry_run
    )

    data_collator = TrOCRDataCollator(processor=processor)

    config.logger.info(f"Training samples: {len(train_dataset)}")
    config.logger.info(f"Validation samples: {len(val_dataset)}")

    # Setup Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_steps=config.warmup_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit = config.save_total_limit, # added
        logging_steps=config.logging_steps,
        predict_with_generate=True,  # Critical for text generation
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer",  
        greater_is_better=False,
        report_to="wandb" if args.wandb else None,
        run_name=f"trocr-swedish-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        remove_unused_columns=False,  # Important for custom datasets
    )

    # Initialize WandB if requested (Seq2SeqTrainer hanterar resten)
    if args.wandb:
        wandb.init(
            project=args.project_name,
            config={
                "model_name": config.model_name,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "dry_run": args.dry_run,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "fp16": config.fp16,
            }
        )

    # Setup Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor,  
        compute_metrics=create_compute_metrics(processor.tokenizer)
    )

    # Train with automatic optimizations
    config.logger.info("Starting training with Seq2SeqTrainer...")
    config.logger.info(f"Training samples: {len(train_dataset)}")
    config.logger.info(f"Validation samples: {len(val_dataset)}")
    
    trainer.train()
    
    # Save final model in multiple formats for compatibility
    final_model_path = config.output_dir / "final_model"
    config.logger.info(f"Saving model in multiple formats to {final_model_path}")

    # Save as pytorch_model.bin (broad compatibility)
    trainer.save_model(str(final_model_path))
    trainer.model.save_pretrained(str(final_model_path), safe_serialization=False)
    config.logger.info("✓ Saved pytorch_model.bin format")

    # Save as safetensors (modern secure format)
    safetensors_path = final_model_path / "safetensors_version"
    trainer.model.save_pretrained(str(safetensors_path), safe_serialization=True)
    config.logger.info("✓ Saved safetensors format")

    # Save processor (compatible with both formats)
    processor.save_pretrained(str(final_model_path))
    config.logger.info("✓ Saved processor configuration")

    config.logger.info(f"Final model saved in both formats to {final_model_path}")
    config.logger.info(f"Primary format: {final_model_path}")
    config.logger.info(f"Safetensors format: {safetensors_path}")

    # Finish WandB
    if args.wandb:
        wandb.finish()

def main():
    # Environment setup
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] ='0'

    args = parse_args()
    train_model(args)

if __name__ == "__main__":
    main()