"""
EUR/USD Forex Fine-Tuning Trainer with LoRA

Multi-task trainer for EUR/USD direction and bucket prediction:
- Uses LoRA for parameter-efficient fine-tuning
- Multi-task loss (direction + bucket)
- Tracks both accuracy and MAE metrics
"""

import os
import logging
from typing import Dict, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ForexDataset(Dataset):
    """Dataset for EUR/USD forex prediction."""
    
    def __init__(
        self,
        texts: List[str],
        direction_labels: np.ndarray,
        bucket_labels: np.ndarray,
        tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of formatted text sequences
            direction_labels: Direction labels (0=down, 1=up)
            bucket_labels: Bucket labels (0-4)
            tokenizer: HuggingFace tokenizer
            max_length: Max sequence length
        """
        self.texts = texts
        self.direction_labels = direction_labels
        self.bucket_labels = bucket_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'direction_label': torch.tensor(self.direction_labels[idx], dtype=torch.long),
            'bucket_label': torch.tensor(self.bucket_labels[idx], dtype=torch.long),
        }


class ForexFineTuneTrainer:
    """
    Trainer for fine-tuning FinBERT on EUR/USD forex prediction.
    
    Uses LoRA for efficient fine-tuning and multi-task learning.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: ForexDataset,
        val_dataset: ForexDataset,
        output_dir: str = "data/models/finbert_forex",
        device: str = "mps",
        # Training hyperparameters
        epochs: int = 30,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        # Loss weighting
        direction_weight: float = 0.5,
        bucket_weight: float = 0.5,
        # LoRA config
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        """
        Initialize trainer.
        
        Args:
            model: FinBERTForexPredictor model
            tokenizer: HuggingFace tokenizer
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save checkpoints
            device: Device to train on
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_ratio: Warmup ratio (fraction of steps)
            direction_weight: Weight for direction loss
            bucket_weight: Weight for bucket loss
            use_lora: Whether to use LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.device = device
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        
        self.direction_weight = direction_weight
        self.bucket_weight = bucket_weight
        
        # Setup LoRA if requested
        if use_lora:
            self.model = self._setup_lora(lora_r, lora_alpha)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Loss functions
        self.direction_criterion = nn.CrossEntropyLoss()
        self.bucket_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'direction_acc': [],
            'bucket_acc': [],
            'bucket_mae': [],
        }
        
        # Early stopping
        self.early_stopping_patience = 3
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ForexFineTuneTrainer initialized")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - LoRA: {use_lora}")
        logger.info(f"  - Early stopping patience: {self.early_stopping_patience}")
        logger.info(f"  - Device: {device}")
    
    def _setup_lora(self, r: int = 16, alpha: int = 32):
        """
        Setup LoRA for parameter-efficient fine-tuning.
        
        Args:
            r: LoRA rank
            alpha: LoRA alpha
            
        Returns:
            Model with LoRA applied
        """
        logger.info(f"Setting up LoRA (r={r}, alpha={alpha})...")
        
        # LoRA configuration - target all attention layers
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["query", "key", "value", "dense"],  # All attention layers
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Apply LoRA to FinBERT only (not the classification heads)
        # Wrap just the FinBERT encoder
        self.model.finbert = get_peft_model(self.model.finbert, lora_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"  ✓ LoRA configured")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        
        return self.model
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            direction_labels = batch['direction_label'].to(self.device)
            bucket_labels = batch['bucket_label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate losses
            direction_loss = self.direction_criterion(
                outputs['direction_logits'],
                direction_labels
            )
            bucket_loss = self.bucket_criterion(
                outputs['bucket_logits'],
                bucket_labels
            )
            
            # Combined loss
            loss = (self.direction_weight * direction_loss + 
                   self.bucket_weight * bucket_loss)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        
        all_direction_preds = []
        all_direction_labels = []
        all_bucket_preds = []
        all_bucket_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                direction_labels = batch['direction_label'].to(self.device)
                bucket_labels = batch['bucket_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate losses
                direction_loss = self.direction_criterion(
                    outputs['direction_logits'],
                    direction_labels
                )
                bucket_loss = self.bucket_criterion(
                    outputs['bucket_logits'],
                    bucket_labels
                )
                
                loss = (self.direction_weight * direction_loss + 
                       self.bucket_weight * bucket_loss)
                
                total_loss += loss.item()
                
                # Get predictions
                direction_preds = torch.argmax(outputs['direction_logits'], dim=1)
                bucket_preds = torch.argmax(outputs['bucket_logits'], dim=1)
                
                all_direction_preds.extend(direction_preds.cpu().numpy())
                all_direction_labels.extend(direction_labels.cpu().numpy())
                all_bucket_preds.extend(bucket_preds.cpu().numpy())
                all_bucket_labels.extend(bucket_labels.cpu().numpy())
        
        # Calculate metrics
        direction_acc = np.mean(np.array(all_direction_preds) == np.array(all_direction_labels))
        bucket_acc = np.mean(np.array(all_bucket_preds) == np.array(all_bucket_labels))
        bucket_mae = np.mean(np.abs(np.array(all_bucket_preds) - np.array(all_bucket_labels)))
        
        avg_loss = total_loss / len(self.val_loader)
        
        return {
            'val_loss': avg_loss,
            'direction_acc': direction_acc,
            'bucket_acc': bucket_acc,
            'bucket_mae': bucket_mae,
        }
    
    def train(self):
        """Complete training loop."""
        logger.info("="*60)
        logger.info(f"Starting EUR/USD Forex Fine-Tuning")
        logger.info("="*60)
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        
        best_val_loss = float('inf')
        best_direction_acc = 0.0
        
        for epoch in range(1, self.epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Log metrics
            logger.info(f"\nEpoch {epoch} Results:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Direction Accuracy: {val_metrics['direction_acc']:.1%}")
            logger.info(f"  Bucket Accuracy: {val_metrics['bucket_acc']:.1%}")
            logger.info(f"  Bucket MAE: {val_metrics['bucket_mae']:.3f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['direction_acc'].append(val_metrics['direction_acc'])
            self.history['bucket_acc'].append(val_metrics['bucket_acc'])
            self.history['bucket_mae'].append(val_metrics['bucket_mae'])
            
            # Save best model
            if val_metrics['direction_acc'] > best_direction_acc:
                best_direction_acc = val_metrics['direction_acc']
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"  ✓ New best model saved (direction acc: {best_direction_acc:.1%})")
            
            # Early stopping check
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.info(f"  ⚠ No improvement for {self.patience_counter}/{self.early_stopping_patience} epochs")
                
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"  🛑 Early stopping triggered at epoch {epoch}")
                    self.early_stopped = True
                    break
        
        logger.info("\n" + "="*60)
        logger.info("Training Complete!")
        logger.info("="*60)
        logger.info(f"Best Direction Accuracy: {best_direction_acc:.1%}")
        logger.info(f"Best Val Loss: {best_val_loss:.4f}")
        if self.early_stopped:
            logger.info(f"Early stopped at epoch {epoch} (patience={self.early_stopping_patience})")
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if is_best:
            save_dir = self.output_dir / "best"
        else:
            save_dir = self.output_dir / f"checkpoint-epoch-{epoch}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
        }, save_dir / 'model.pt')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        logger.debug(f"  Checkpoint saved to {save_dir}")
    
    def save_history(self):
        """Save training history to JSON."""
        import json
        
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"  ✓ Training history saved to {history_path}")


def setup_lora_model(
    model,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: List[str] = ["query", "key", "value", "dense"]
):
    """
    Setup LoRA configuration for the model.
    
    Args:
        model: Model to apply LoRA to
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        target_modules: Modules to apply LoRA to
        
    Returns:
        Model with LoRA applied
    """
    logger.info(f"Configuring LoRA (r={lora_r}, alpha={lora_alpha})...")
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    # Apply LoRA only to FinBERT encoder (not classification heads)
    model.finbert = get_peft_model(model.finbert, lora_config)
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"  ✓ LoRA applied to FinBERT")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable with LoRA: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model


def test_trainer():
    """Test the fine-tuning trainer setup."""
    logging.basicConfig(level=logging.INFO)
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.models.finbert.finetune_head import FinBERTForexPredictor
    from src.preprocessing.forex_preprocessor import ForexDataPreprocessor
    
    # Load data
    logger.info("Loading EUR/USD dataset...")
    train_df = pd.read_parquet('data/processed/train.parquet')[:100]  # Small sample
    val_df = pd.read_parquet('data/processed/validation.parquet')[:20]
    
    # Preprocess
    preprocessor = ForexDataPreprocessor()
    train_processed, train_texts = preprocessor.prepare_dataset(train_df, fit=True)
    val_processed, val_texts = preprocessor.prepare_dataset(val_df, fit=False)
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    train_dataset = ForexDataset(
        texts=train_texts,
        direction_labels=train_processed['direction_label'].values,
        bucket_labels=train_processed['bucket_label'].values,
        tokenizer=tokenizer,
    )
    
    val_dataset = ForexDataset(
        texts=val_texts,
        direction_labels=val_processed['direction_label'].values,
        bucket_labels=val_processed['bucket_label'].values,
        tokenizer=tokenizer,
    )
    
    # Create model
    model = FinBERTForexPredictor()
    
    # Apply LoRA
    model = setup_lora_model(model, lora_r=16, lora_alpha=32)
    
    # Create trainer
    trainer = ForexFineTuneTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=2,  # Just test 2 epochs
        batch_size=8,
        use_lora=False,  # Already applied manually
    )
    
    print("\n" + "="*60)
    print("Forex Fine-Tuning Trainer Test")
    print("="*60)
    print(f"✓ Trainer initialized")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples: {len(val_dataset)}")
    print(f"  - Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test one training step
    logger.info("\nTesting one training step...")
    train_loss = trainer.train_epoch()
    val_metrics = trainer.evaluate()
    
    print(f"\nTest training step complete:")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val loss: {val_metrics['val_loss']:.4f}")
    print(f"  Direction acc: {val_metrics['direction_acc']:.1%}")
    print(f"  Bucket acc: {val_metrics['bucket_acc']:.1%}")
    
    print("\n✓ Trainer test successful!")


if __name__ == "__main__":
    test_trainer()

