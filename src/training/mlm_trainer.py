"""
Masked Language Modeling (MLM) Trainer for SSL Pre-training

This module implements MLM training for Gemma and other language models
using the financial news + price sequences dataset.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLMTrainer:
    """
    Trainer for Masked Language Modeling (MLM) on financial sequences.
    
    This trainer:
    - Loads pre-processed financial news + price sequences
    - Tokenizes sequences with the model's tokenizer
    - Applies MLM masking (15% of tokens)
    - Trains using HuggingFace Trainer API
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str,
        config: Dict[str, Any],
    ):
        """
        Initialize MLM Trainer.
        
        Args:
            model: Pre-trained language model
            tokenizer: Tokenizer for the model
            device: Device to train on ("mps", "cuda", or "cpu")
            config: Training configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Data collator for MLM
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Set to False for causal LM (Gemma)
            mlm_probability=config.get("mlm_probability", 0.15),
        )
        
        logger.info("MLMTrainer initialized")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - MLM probability: {config.get('mlm_probability', 0.15)}")
    
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
            examples: Batch of examples with 'sequence' field
            
        Returns:
            Tokenized examples
        """
        # Tokenize the sequences
        tokenized = self.tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=self.config.get("max_seq_length", 512),
            return_tensors=None,  # Return lists, not tensors
        )
        
        # For causal LM, labels are same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """
        Prepare dataset by tokenizing sequences.
        
        Args:
            dataset: Raw dataset
            
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
        
        return tokenized_dataset
    
    def create_training_arguments(
        self,
        output_dir: str,
        num_epochs: int = 2,
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
            per_device_train_batch_size=self.config.get("batch_size", 4),
            per_device_eval_batch_size=self.config.get("batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 2),
            learning_rate=self.config.get("learning_rate", 1e-5),
            warmup_steps=self.config.get("warmup_steps", 100),
            
            # Optimizer settings
            optim="adamw_torch",
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            
            # Checkpointing
            save_strategy="steps",
            save_steps=self.config.get("save_steps", 500),
            save_total_limit=self.config.get("save_total_limit", 3),
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=self.config.get("evaluation_steps", 100),
            
            # Logging
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=self.config.get("logging_steps", 10),
            
            # Device settings
            use_cpu=self.device == "cpu",
            # Note: use_mps_device is deprecated, torch will auto-detect
            
            # Memory optimization
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            fp16=False,  # MPS doesn't support fp16 training yet
            
            # Other settings
            remove_unused_columns=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Reproducibility
            seed=42,
            data_seed=42,
        )
        
        logger.info("Training arguments created")
        logger.info(f"  - Output dir: {output_dir}")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Batch size: {self.config.get('batch_size', 4)}")
        logger.info(f"  - Effective batch size: {self.config.get('batch_size', 4) * self.config.get('gradient_accumulation_steps', 2)}")
        logger.info(f"  - Learning rate: {self.config.get('learning_rate', 1e-5)}")
        
        return training_args
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "data/models/gemma",
        num_epochs: int = 2,
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
        logger.info("Starting MLM Training")
        logger.info("=" * 70)
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset)
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
        
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info(f"  - Final loss: {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"  - Training time: {metrics.get('train_runtime', 'N/A'):.2f}s")
        logger.info("=" * 70)
        
        return metrics


def train_gemma_mlm(
    model_name: str = "google/gemma-3-270m",
    dataset_path: str = "data/processed",
    output_dir: str = "data/models/gemma",
    num_epochs: int = 2,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    use_auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to train Gemma with MLM from scratch.
    
    Args:
        model_name: HuggingFace model name
        dataset_path: Path to preprocessed dataset
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_auth_token: HuggingFace token
        
    Returns:
        Training metrics
    """
    # Import here to avoid circular imports
    from src.models.gemma import load_gemma_model, load_gemma_tokenizer, GemmaConfig
    
    # Create config
    config = GemmaConfig(
        model_name=model_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        checkpoint_dir=output_dir,
    )
    
    # Load model and tokenizer
    tokenizer = load_gemma_tokenizer(config, use_auth_token)
    model, device = load_gemma_model(config, use_auth_token)
    
    # Create trainer
    trainer = MLMTrainer(
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

