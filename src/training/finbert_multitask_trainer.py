"""
Multi-task Trainer for FinBERT SSL Pre-training

This module implements training for FinBERT with combined objectives:
- Masked Language Modeling (MLM)
- Contrastive Learning

Supports configurable loss weighting, learning rate scheduling, and comprehensive metrics tracking.
"""

import os
import logging
import math
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
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm

from .contrastive_loss import NTXentLoss

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiTaskTrainer:
    """
    Trainer for multi-task SSL pre-training of FinBERT.
    
    Combines:
    - MLM (Masked Language Modeling) for language understanding
    - Contrastive learning for representation learning
    
    Features:
    - Configurable loss weighting (alpha * MLM + (1-alpha) * Contrastive)
    - Learning rate scheduling (cosine with warmup)
    - Perplexity tracking for MLM
    - Best checkpoint saving based on combined validation metrics
    """
    
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        device: str,
        config: Dict[str, Any],
    ):
        """
        Initialize Multi-task Trainer.
        
        Args:
            model: Multi-task FinBERT model
            tokenizer: FinBERT tokenizer
            device: Device to train on ("mps", "cuda", or "cpu")
            config: Training configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # Data collator for MLM (handles masking)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=config.get("mlm_probability", 0.15),
        )
        
        # Contrastive loss
        self.contrastive_loss_fn = NTXentLoss(
            temperature=config.get("temperature", 0.07)
        )
        
        # Loss weighting
        self.mlm_weight = config.get("mlm_weight", 0.5)
        self.contrastive_weight = config.get("contrastive_weight", 0.5)
        
        # Verify weights sum to 1
        total_weight = self.mlm_weight + self.contrastive_weight
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"Loss weights sum to {total_weight}, normalizing...")
            self.mlm_weight /= total_weight
            self.contrastive_weight /= total_weight
        
        logger.info("MultiTaskTrainer initialized")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - MLM weight: {self.mlm_weight:.2f}")
        logger.info(f"  - Contrastive weight: {self.contrastive_weight:.2f}")
        logger.info(f"  - MLM probability: {config.get('mlm_probability', 0.15)}")
        logger.info(f"  - Temperature: {config.get('temperature', 0.07)}")
    
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
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize sequences.
        
        Args:
            examples: Batch of examples with 'sequence_text' field
            
        Returns:
            Tokenized examples
        """
        tokenized = self.tokenizer(
            examples["sequence_text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.get("max_seq_length", 512),
            return_tensors=None,
        )
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
        
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing sequences",
        )
        
        logger.info(f"✓ Tokenization complete: {len(tokenized_dataset)} samples")
        
        return tokenized_dataset
    
    def create_optimizer(self, num_training_steps: int):
        """
        Create optimizer and learning rate scheduler.
        
        Args:
            num_training_steps: Total number of training steps
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Cosine learning rate scheduler with warmup
        warmup_steps = int(num_training_steps * self.config.get("warmup_ratio", 0.1))
        
        def lr_lambda(current_step: int):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # Cosine decay phase
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        logger.info("Optimizer and scheduler created")
        logger.info(f"  - Learning rate: {self.config.get('learning_rate', 1e-5)}")
        logger.info(f"  - Warmup steps: {warmup_steps} ({warmup_steps/num_training_steps*100:.1f}%)")
        logger.info(f"  - Total steps: {num_training_steps}")
        logger.info(f"  - Scheduler: Cosine with warmup")
        
        return optimizer, scheduler
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of tokenized data
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_mlm_loss=True,
            return_embeddings=True,
        )
        
        # MLM loss
        mlm_loss = outputs['mlm_loss']
        
        # Contrastive loss
        embeddings = outputs['embeddings']
        if len(embeddings) >= 2:
            # Split batch for contrastive pairs
            mid_point = len(embeddings) // 2
            embeddings_a = embeddings[:mid_point]
            embeddings_b = embeddings[mid_point:2*mid_point]
            contrastive_loss = self.contrastive_loss_fn(embeddings_a, embeddings_b)
        else:
            contrastive_loss = torch.tensor(0.0, device=self.device)
        
        # Combined loss
        total_loss = (
            self.mlm_weight * mlm_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'mlm_loss': mlm_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'lr': scheduler.get_last_lr()[0],
        }
    
    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Args:
            dataset: Validation dataset (tokenized)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_mlm_loss = 0.0
        total_contrastive_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        batch_size = self.config.get("batch_size", 8)
        num_samples = len(dataset)
        
        # Create batches
        with torch.no_grad():
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_indices = list(range(batch_start, batch_end))
                
                # Get batch
                batch_data = [dataset[i] for i in batch_indices]
                
                # Collate batch (apply MLM masking)
                batch = self.data_collator(batch_data)
                
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_mlm_loss=True,
                    return_embeddings=True,
                )
                
                # MLM loss
                mlm_loss = outputs['mlm_loss']
                
                # Contrastive loss
                embeddings = outputs['embeddings']
                if len(embeddings) >= 2:
                    mid_point = len(embeddings) // 2
                    embeddings_a = embeddings[:mid_point]
                    embeddings_b = embeddings[mid_point:2*mid_point]
                    contrastive_loss = self.contrastive_loss_fn(embeddings_a, embeddings_b)
                else:
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                
                # Combined loss
                combined_loss = (
                    self.mlm_weight * mlm_loss +
                    self.contrastive_weight * contrastive_loss
                )
                
                total_mlm_loss += mlm_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                total_loss += combined_loss.item()
                num_batches += 1
        
        # Average losses
        avg_mlm_loss = total_mlm_loss / max(num_batches, 1)
        avg_contrastive_loss = total_contrastive_loss / max(num_batches, 1)
        avg_total_loss = total_loss / max(num_batches, 1)
        
        # Compute perplexity
        perplexity = math.exp(avg_mlm_loss) if avg_mlm_loss < 20 else float('inf')
        
        return {
            'total_loss': avg_total_loss,
            'mlm_loss': avg_mlm_loss,
            'contrastive_loss': avg_contrastive_loss,
            'perplexity': perplexity,
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "data/models/finbert_multitask",
        num_epochs: int = 20,
    ) -> Dict[str, Any]:
        """
        Train the model with multi-task learning.
        
        Args:
            train_dataset: Training dataset (raw, will be tokenized)
            eval_dataset: Evaluation dataset (raw, will be tokenized)
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            
        Returns:
            Training metrics dictionary
        """
        logger.info("=" * 80)
        logger.info("Starting Multi-Task SSL Pre-training (FinBERT)")
        logger.info("=" * 80)
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare datasets
        logger.info("\nPreparing datasets...")
        train_dataset = self.prepare_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)
        
        # Setup training
        batch_size = self.config.get("batch_size", 8)
        num_samples = len(train_dataset)
        steps_per_epoch = num_samples // batch_size
        total_steps = steps_per_epoch * num_epochs
        
        optimizer, scheduler = self.create_optimizer(total_steps)
        
        # Training state
        best_eval_loss = float('inf')
        best_perplexity = float('inf')
        global_step = 0
        
        metrics = {
            'train_losses': [],
            'train_mlm_losses': [],
            'train_contrastive_losses': [],
            'eval_losses': [],
            'eval_mlm_losses': [],
            'eval_contrastive_losses': [],
            'perplexities': [],
            'learning_rates': [],
        }
        
        logger.info("\nTraining configuration:")
        logger.info(f"  - Train samples: {len(train_dataset)}")
        logger.info(f"  - Eval samples: {len(eval_dataset) if eval_dataset else 0}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Steps per epoch: {steps_per_epoch}")
        logger.info(f"  - Total steps: {total_steps}")
        logger.info(f"  - Epochs: {num_epochs}")
        
        # Training loop
        logger.info("\n" + "=" * 80)
        logger.info("Starting Training Loop")
        logger.info("=" * 80 + "\n")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info("-" * 80)
            
            # Training
            epoch_mlm_loss = 0.0
            epoch_contrastive_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            
            # Shuffle indices
            import random
            indices = list(range(num_samples))
            random.shuffle(indices)
            
            # Progress bar
            progress_bar = tqdm(range(0, num_samples, batch_size), desc=f"Training")
            
            for batch_start in progress_bar:
                batch_end = min(batch_start + batch_size, num_samples)
                batch_indices = indices[batch_start:batch_end]
                
                # Get batch
                batch_data = [train_dataset[i] for i in batch_indices]
                
                # Collate batch (apply MLM masking)
                batch = self.data_collator(batch_data)
                
                # Training step
                step_metrics = self.train_step(batch, optimizer, scheduler)
                
                # Accumulate losses
                epoch_total_loss += step_metrics['total_loss']
                epoch_mlm_loss += step_metrics['mlm_loss']
                epoch_contrastive_loss += step_metrics['contrastive_loss']
                num_batches += 1
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_metrics['total_loss']:.4f}",
                    'mlm': f"{step_metrics['mlm_loss']:.4f}",
                    'cont': f"{step_metrics['contrastive_loss']:.4f}",
                    'lr': f"{step_metrics['lr']:.2e}",
                })
            
            # Epoch metrics
            avg_total_loss = epoch_total_loss / num_batches
            avg_mlm_loss = epoch_mlm_loss / num_batches
            avg_contrastive_loss = epoch_contrastive_loss / num_batches
            
            metrics['train_losses'].append(avg_total_loss)
            metrics['train_mlm_losses'].append(avg_mlm_loss)
            metrics['train_contrastive_losses'].append(avg_contrastive_loss)
            metrics['learning_rates'].append(scheduler.get_last_lr()[0])
            
            logger.info(f"\nEpoch {epoch} Training Results:")
            logger.info(f"  - Total loss: {avg_total_loss:.4f}")
            logger.info(f"  - MLM loss: {avg_mlm_loss:.4f}")
            logger.info(f"  - Contrastive loss: {avg_contrastive_loss:.4f}")
            
            # Evaluation
            if eval_dataset is not None:
                logger.info("\nEvaluating...")
                eval_metrics = self.evaluate(eval_dataset)
                
                metrics['eval_losses'].append(eval_metrics['total_loss'])
                metrics['eval_mlm_losses'].append(eval_metrics['mlm_loss'])
                metrics['eval_contrastive_losses'].append(eval_metrics['contrastive_loss'])
                metrics['perplexities'].append(eval_metrics['perplexity'])
                
                logger.info(f"\nEpoch {epoch} Validation Results:")
                logger.info(f"  - Total loss: {eval_metrics['total_loss']:.4f}")
                logger.info(f"  - MLM loss: {eval_metrics['mlm_loss']:.4f}")
                logger.info(f"  - Perplexity: {eval_metrics['perplexity']:.4f}")
                logger.info(f"  - Contrastive loss: {eval_metrics['contrastive_loss']:.4f}")
                
                # Save best model
                if eval_metrics['total_loss'] < best_eval_loss:
                    best_eval_loss = eval_metrics['total_loss']
                    best_perplexity = eval_metrics['perplexity']
                    logger.info(f"\n✓ New best model! Loss: {best_eval_loss:.4f}, Perplexity: {best_perplexity:.4f}")
                    self.save_model(output_dir, is_best=True)
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
                self.save_model(checkpoint_dir, is_best=False)
                logger.info(f"\n✓ Checkpoint saved: {checkpoint_dir}")
            
            logger.info("\n" + "=" * 80 + "\n")
        
        # Save final model
        final_dir = os.path.join(output_dir, "final")
        self.save_model(final_dir, is_best=False)
        
        # Training complete
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"  - Best validation loss: {best_eval_loss:.4f}")
        logger.info(f"  - Best perplexity: {best_perplexity:.4f}")
        logger.info(f"  - Final train loss: {metrics['train_losses'][-1]:.4f}")
        logger.info(f"  - Total epochs: {num_epochs}")
        logger.info(f"  - Total steps: {global_step}")
        logger.info("=" * 80)
        
        # Save metrics
        self._save_metrics(metrics, output_dir)
        
        return metrics
    
    def save_model(self, output_dir: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            output_dir: Directory to save model
            is_best: Whether this is the best checkpoint
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'is_best': is_best,
        }
        
        torch.save(checkpoint, os.path.join(output_dir, 'multitask_model.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def _save_metrics(self, metrics: Dict[str, Any], output_dir: str):
        """Save training metrics to JSON."""
        import json
        
        metrics_file = os.path.join(output_dir, 'training_metrics.json')
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")

