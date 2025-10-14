"""
Contrastive Learning Trainer for FinBERT

This module implements contrastive learning training for FinBERT
on financial news + price sequences.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Disable TensorFlow imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = 'NO'

import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from tqdm import tqdm

from .contrastive_loss import NTXentLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    """
    Trainer for contrastive learning on financial sequences using FinBERT.
    
    This trainer:
    - Loads pre-trained FinBERT with projection head
    - Creates positive/negative pairs from news + price data
    - Trains using NT-Xent (InfoNCE) loss
    """
    
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: str,
        config: Dict[str, Any],
    ):
        """
        Initialize Contrastive Trainer.
        
        Args:
            model: FinBERT model with projection head
            tokenizer: FinBERT tokenizer
            device: Device to train on ("mps", "cuda", or "cpu")
            config: Training configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Contrastive loss
        self.loss_fn = NTXentLoss(
            temperature=config.get("temperature", 0.07)
        )
        
        logger.info("ContrastiveTrainer initialized")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Temperature: {config.get('temperature', 0.07)}")
        logger.info(f"  - Loss function: NT-Xent (InfoNCE)")
    
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
        
        file_path = os.path.join(dataset_path, f"{split}.parquet")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        dataset = load_dataset("parquet", data_files=file_path, split="train")
        
        logger.info(f"✓ Loaded {len(dataset)} samples")
        logger.info(f"  - Columns: {dataset.column_names}")
        
        return dataset
    
    def tokenize_sequences(self, sequences: list) -> Dict[str, torch.Tensor]:
        """
        Tokenize sequences.
        
        Args:
            sequences: List of sequence texts
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        tokenized = self.tokenizer(
            sequences,
            truncation=True,
            padding="max_length",
            max_length=self.config.get("max_seq_length", 512),
            return_tensors="pt",
        )
        return tokenized
    
    def train_epoch(
        self,
        dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataset: Training dataset
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        batch_size = self.config.get("batch_size", 8)
        
        # Create batches
        num_samples = len(dataset)
        indices = list(range(num_samples))
        
        # Shuffle indices
        import random
        random.shuffle(indices)
        
        # Process in batches
        progress_bar = tqdm(range(0, num_samples, batch_size), desc=f"Epoch {epoch}")
        
        for batch_start in progress_bar:
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]
            
            # Get batch data
            batch_sequences = [dataset[i]['sequence_text'] for i in batch_indices]
            
            # Tokenize
            tokenized = self.tokenize_sequences(batch_sequences)
            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)
            
            # Forward pass - get embeddings
            embeddings = self.model(input_ids, attention_mask)
            
            # For NT-Xent, we treat the batch as creating pairs within itself
            # Split batch in half to create positive pairs
            if len(embeddings) >= 2:
                mid_point = len(embeddings) // 2
                embeddings_a = embeddings[:mid_point]
                embeddings_b = embeddings[mid_point:2*mid_point]
                
                # Compute NT-Xent loss
                loss = self.loss_fn(embeddings_a, embeddings_b)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def evaluate(self, dataset: Dataset) -> float:
        """
        Evaluate on validation set.
        
        Args:
            dataset: Validation dataset
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        batch_size = self.config.get("batch_size", 8)
        num_samples = len(dataset)
        
        with torch.no_grad():
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                
                # Get batch data
                batch_sequences = [dataset[i]['sequence_text'] for i in range(batch_start, batch_end)]
                
                # Tokenize
                tokenized = self.tokenize_sequences(batch_sequences)
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)
                
                # Forward pass
                embeddings = self.model(input_ids, attention_mask)
                
                # Compute loss (same as training)
                if len(embeddings) >= 2:
                    mid_point = len(embeddings) // 2
                    embeddings_a = embeddings[:mid_point]
                    embeddings_b = embeddings[mid_point:2*mid_point]
                    
                    loss = self.loss_fn(embeddings_a, embeddings_b)
                    total_loss += loss.item()
                    num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "data/models/finbert_contrastive",
        num_epochs: int = 10,
        learning_rate: float = 1e-5,
    ) -> Dict[str, Any]:
        """
        Train the model using contrastive learning.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training metrics dictionary
        """
        logger.info("=" * 70)
        logger.info("Starting Contrastive Learning Training")
        logger.info("=" * 70)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Training loop
        best_eval_loss = float('inf')
        metrics = {
            'train_losses': [],
            'eval_losses': [],
        }
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_dataset, optimizer, epoch)
            metrics['train_losses'].append(train_loss)
            logger.info(f"Train loss: {train_loss:.4f}")
            
            # Evaluate
            if eval_dataset is not None:
                eval_loss = self.evaluate(eval_dataset)
                metrics['eval_losses'].append(eval_loss)
                logger.info(f"Eval loss: {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    logger.info(f"✓ New best model! Eval loss: {eval_loss:.4f}")
                    self.save_model(output_dir)
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
                self.save_model(checkpoint_dir)
                logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
        
        # Save final model
        self.save_model(output_dir)
        
        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info(f"  - Best eval loss: {best_eval_loss:.4f}")
        logger.info(f"  - Final train loss: {metrics['train_losses'][-1]:.4f}")
        logger.info("=" * 70)
        
        return metrics
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, os.path.join(output_dir, 'contrastive_model.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")

