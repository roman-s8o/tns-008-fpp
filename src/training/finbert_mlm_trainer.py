"""
FinBERT MLM Trainer for SSL Pre-training

This module implements Masked Language Modeling (MLM) training for FinBERT
using the financial news + price sequences dataset.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Disable TensorFlow imports (we only use PyTorch)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

import torch
from datasets import load_dataset, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinBERTMLMTrainer:
    """
    Trainer for Masked Language Modeling (MLM) on financial sequences using FinBERT.
    
    This trainer:
    - Loads pre-processed financial news + price sequences
    - Tokenizes sequences with FinBERT tokenizer
    - Applies MLM masking (15% of tokens: 80% [MASK], 10% random, 10% original)
    - Trains using HuggingFace Trainer API
    """
    
    def __init__(
        self,
        model: AutoModelForMaskedLM,
        tokenizer: AutoTokenizer,
        device: str,
        config: Dict[str, Any],
    ):
        """
        Initialize FinBERT MLM Trainer.
        
        Args:
            model: FinBERT model for MLM
            tokenizer: FinBERT tokenizer
            device: Device to train on ("mps", "cuda", or "cpu")
            config: Training configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Data collator for MLM (handles masking automatically)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,  # Enable MLM
            mlm_probability=config.get("mlm_probability", 0.15),
        )
        
        logger.info("FinBERTMLMTrainer initialized")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - MLM probability: {config.get('mlm_probability', 0.15)}")
        logger.info(f"  - Masking strategy: 80% [MASK], 10% random, 10% original")
    
    def load_dataset(self, dataset_path: str, split: str = "train") -> Dataset:
        """
        Load preprocessed dataset from parquet files.
        
        Args:
            dataset_path: Path to dataset directory
            split: Dataset split ("train", "validation", or "finetune")
            
        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Loading {split} dataset from {dataset_path}")
        
        # Load from parquet
        file_path = os.path.join(dataset_path, f"{split}.parquet")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        dataset = load_dataset("parquet", data_files=file_path, split="train")
        
        logger.info(f"✓ Loaded {len(dataset)} samples")
        logger.info(f"  - Columns: {dataset.column_names}")
        
        return dataset
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize sequences for MLM training.
        
        Args:
            examples: Batch of examples with 'sequence_text' field
            
        Returns:
            Tokenized examples
        """
        # Tokenize the sequences (column name is sequence_text in the dataset)
        tokenized = self.tokenizer(
            examples["sequence_text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.get("max_seq_length", 512),
            return_tensors=None,  # Return lists, not tensors
        )
        
        return tokenized
    
    def prepare_dataset(self, dataset: Dataset, save_sample: bool = False) -> Dataset:
        """
        Prepare dataset by tokenizing sequences.
        
        Args:
            dataset: Raw dataset
            save_sample: If True, save a sample of tokenized data
            
        Returns:
            Tokenized dataset
        """
        logger.info("Tokenizing dataset...")
        
        # Tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing sequences",
        )
        
        logger.info(f"✓ Tokenization complete")
        logger.info(f"  - Total samples: {len(tokenized_dataset)}")
        
        # Save a sample if requested
        if save_sample and len(tokenized_dataset) > 0:
            self._save_tokenized_sample(tokenized_dataset)
        
        return tokenized_dataset
    
    def _save_tokenized_sample(self, dataset: Dataset, num_samples: int = 5):
        """
        Save a sample of tokenized data for inspection.
        
        Args:
            dataset: Tokenized dataset
            num_samples: Number of samples to save
        """
        output_dir = Path(self.config.get("checkpoint_dir", "data/models/finbert"))
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_file = output_dir / "tokenized_sample.txt"
        
        with open(sample_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FinBERT Tokenized Samples\n")
            f.write("=" * 80 + "\n\n")
            
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                
                f.write(f"Sample {i+1}:\n")
                f.write("-" * 80 + "\n")
                
                # Decode tokens
                tokens = sample['input_ids']
                decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=False)
                
                f.write(f"Token IDs (first 50): {tokens[:50]}\n")
                f.write(f"Length: {len(tokens)} tokens\n")
                f.write(f"\nDecoded text:\n{decoded_text}\n")
                f.write("\n" + "=" * 80 + "\n\n")
        
        logger.info(f"✓ Tokenized samples saved to: {sample_file}")
    
    def create_training_arguments(
        self,
        output_dir: str,
        num_epochs: int = 3,
    ) -> TrainingArguments:
        """
        Create training arguments for HuggingFace Trainer.
        
        Args:
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            
        Returns:
            TrainingArguments object
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Note: For MPS, we need to use specific settings
        use_mps = self.device == "mps"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Training hyperparameters
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.config.get("batch_size", 16),
            per_device_eval_batch_size=self.config.get("batch_size", 16),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            learning_rate=self.config.get("learning_rate", 2e-5),
            warmup_steps=self.config.get("warmup_steps", 100),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            
            # Optimizer settings
            optim="adamw_torch",
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            
            # Checkpointing
            save_strategy="steps",
            save_steps=self.config.get("save_steps", 100),
            save_total_limit=self.config.get("save_total_limit", 3),
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=self.config.get("evaluation_steps", 50),
            
            # Logging
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=self.config.get("logging_steps", 10),
            report_to=["tensorboard"],
            
            # Device settings
            use_cpu=self.device == "cpu",
            dataloader_num_workers=self.config.get("dataloader_num_workers", 0),
            
            # Memory optimization
            gradient_checkpointing=self.config.get("gradient_checkpointing", False),
            fp16=False,  # MPS doesn't support fp16 training yet
            
            # Other settings
            remove_unused_columns=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Reproducibility
            seed=42,
        )
        
        logger.info("Training arguments created")
        logger.info(f"  - Output dir: {output_dir}")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Batch size: {self.config.get('batch_size', 16)}")
        logger.info(f"  - Effective batch size: {self.config.get('batch_size', 16) * self.config.get('gradient_accumulation_steps', 1)}")
        logger.info(f"  - Learning rate: {self.config.get('learning_rate', 2e-5)}")
        
        return training_args
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "data/models/finbert",
        num_epochs: int = 3,
    ) -> Dict[str, Any]:
        """
        Train the model using MLM objective.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            
        Returns:
            Training metrics dictionary
        """
        logger.info("=" * 70)
        logger.info("Starting FinBERT MLM Training")
        logger.info("=" * 70)
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset, save_sample=True)
        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)
        
        # Create training arguments
        training_args = self.create_training_arguments(output_dir, num_epochs)
        
        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)
        
        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate on validation set if provided
        if eval_dataset is not None:
            logger.info("Evaluating on validation set...")
            eval_metrics = trainer.evaluate()
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
            metrics.update({"eval_" + k: v for k, v in eval_metrics.items()})
        
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info(f"  - Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        if eval_dataset is not None:
            logger.info(f"  - Final eval loss: {metrics.get('eval_eval_loss', metrics.get('eval_loss', 'N/A')):.4f}")
        logger.info(f"  - Training time: {metrics.get('train_runtime', 'N/A'):.2f}s")
        logger.info(f"  - Samples/second: {metrics.get('train_samples_per_second', 'N/A'):.2f}")
        logger.info("=" * 70)
        
        return metrics


def train_finbert_mlm(
    model_name: str = "ProsusAI/finbert",
    dataset_path: str = "data/processed",
    output_dir: str = "data/models/finbert",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> Dict[str, Any]:
    """
    Convenience function to train FinBERT with MLM from scratch.
    
    Args:
        model_name: HuggingFace model name
        dataset_path: Path to preprocessed dataset
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        
    Returns:
        Training metrics
    """
    # Import here to avoid circular imports
    from src.models.finbert import load_finbert_model, load_finbert_tokenizer, FinBERTConfig
    
    # Create config
    config = FinBERTConfig(
        model_name=model_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        checkpoint_dir=output_dir,
    )
    
    # Load model and tokenizer
    tokenizer = load_finbert_tokenizer(config)
    model, device = load_finbert_model(config)
    
    # Create trainer
    trainer = FinBERTMLMTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=config.__dict__,
    )
    
    # Load datasets
    train_dataset = trainer.load_dataset(dataset_path, split="train")
    eval_dataset = trainer.load_dataset(dataset_path, split="validation")
    
    # Train
    metrics = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        num_epochs=num_epochs,
    )
    
    return metrics

