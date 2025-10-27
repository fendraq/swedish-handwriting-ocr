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
    model_name = 'microsoft/trocr-base-stage1'

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
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    
    # Logging
    parser.add_argument("--wandb", action="store_true",
                       help="Enable WandB experiment tracking")
    parser.add_argument("--project_name", type=str, default="swedish-handwriting-ocr",
                       help="WandB project name")
    
    return parser.parse_args()

def add_swedish_tokens(model, tokenizer, logger):
    """Fix corrupted Swedish tokens and add missing ones"""
    swedish_mappings = {
        "å": ["å", "Ã¥"],  # Correct vs corrupted variants
        "ä": ["ä", "Ã¤"], 
        "ö": ["ö", "Ã¶"],
        "Å": ["Å"],        # Missing entirely
        "Ä": ["Ä"], 
        "Ö": ["Ö"]
    }
    
    logger.info("Analyzing Swedish character support...")

    # Analyze current state
    vocab = tokenizer.get_vocab()
    tokens_to_add = []
    corrupted_tokens = {}

    for correct_char, variants in swedish_mappings.items():
        found_correct = False
        found_corrupted = None

        # Check if correct version exists
        if correct_char in vocab:
            # Verify decoding correctly
            token_id = vocab[correct_char]
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
            if decoded == correct_char:
                logger.info(f"'{correct_char}' correctly supported")
                found_correct = True

        if not found_correct:
            for variant in variants[1:]: # Skip first (correct) variant
                if variant in vocab:
                    logger.warning(f"Found corrupted token: '{variant}' for '{correct_char}'")
                    corrupted_tokens[variant] = correct_char
                    found_corrupted = variant
                    break
        
        # If neither correct nor corrupted found, need to add
        if not found_correct and not found_corrupted:
            tokens_to_add.append(correct_char)
            logger.info(f"'{correct_char}' missing - will add")

    # Handle corrupted tokens by adding correct versions
    if corrupted_tokens:
        logger.info(f"Adding correct version for corrupted tokens: {list(corrupted_tokens.values())}")
        for correct_char in corrupted_tokens.values():
            if correct_char not in tokens_to_add:
                tokens_to_add.append(correct_char)

    # Add missing tokens if any
    if not tokens_to_add:
        logger.info("All Swedish characters properly supported")
        return False
    
    logger.info(f"Adding Swedish tokens: {tokens_to_add}")

    # Save old embedding stat before any changes
    old_embeddings = model.decoder.get_input_embeddings()
    old_vocab_size = old_embeddings.weight.size(0)

    # Add new tokens to tokenizer
    num_added = tokenizer.add_tokens(tokens_to_add)

    if num_added == 0:
        logger.warning("No tokens were added - they might already exist")
        return False

    # Resize model embeddings to match new vocabulary size
    new_vocab_size = len(tokenizer)
    logger.info(f"Resizing embeddings: {old_vocab_size} -> {new_vocab_size}")

    model.decoder.resize_token_embeddings(new_vocab_size)

    # Initialize new token embeddings
    with torch.no_grad():
        new_embeddings = model.decoder.get_input_embeddings()

        # Copy all old embeddings to their original positions
        new_embeddings.weight[:old_vocab_size] = old_embeddings.weight

        # Initialize new tokens with smart values
        if old_vocab_size < new_vocab_size:
            # For corrupted replacements, copy from corrupted token if possible
            mean_embedding = old_embeddings.weight.mean(dim=0, keepdim=True)

            for i, token in enumerate(tokens_to_add):
                new_token_idx = old_vocab_size + i

                # Try to find corrupter version to copy from
                initialized = False
                for corrupted, correct in corrupted_tokens.items():
                    if token == correct and corrupted in vocab:
                        corrupted_idx = vocab[corrupted]
                        if corrupted_idx < old_vocab_size:
                            new_embeddings.weight[new_token_idx] = old_embeddings.weight[corrupted_idx]
                            logger.info(f"Initialized '{token}' from corrupted '{corrupted}'")
                            initialized = True
                            break
                
                if not initialized:
                    new_embeddings.weight[new_token_idx] = mean_embedding.squeeze(0)
                    logger.info(f"Initialized '{token}' with mean embedding")

    logger.info(f"Successfully added {num_added} Swedish tokens and resized embeddings")
    return True
    
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
    
    # Add missing Swedish tokens BEFORE validation
    add_swedish_tokens(model, processor.tokenizer, config.logger)
    
    # Configure model for Swedish handwriting
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # Configure generation settings
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