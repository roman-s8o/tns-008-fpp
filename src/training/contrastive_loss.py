"""
Contrastive Loss Functions for SSL

This module implements various contrastive loss functions including
NT-Xent (InfoNCE) for self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    Also known as InfoNCE loss, commonly used in SimCLR and other
    contrastive learning methods.
    
    For each positive pair (i, j), treats j as the positive example
    and all other samples in the batch as negatives.
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature scaling parameter (default: 0.07)
            reduction: Reduction method ('mean' or 'sum')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            embeddings_a: First set of embeddings (batch_size, embedding_dim)
            embeddings_b: Second set of embeddings (batch_size, embedding_dim)
                         These should be positive pairs with embeddings_a
            
        Returns:
            NT-Xent loss value
        """
        batch_size = embeddings_a.shape[0]
        device = embeddings_a.device
        
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        # Concatenate embeddings: [batch_a, batch_b]
        embeddings = torch.cat([embeddings_a, embeddings_b], dim=0)  # (2*batch_size, dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (2N, 2N)
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create mask for positive pairs
        # Positive pairs are (i, i+N) and (i+N, i)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        
        # Create labels: for each sample i, the positive is at position i+N (or i-N)
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels])  # [N, N+1, ..., 2N-1, 0, 1, ..., N-1]
        
        # Mask out self-similarities
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels, reduction=self.reduction)
        
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for contrastive learning.
    
    Similar to NT-Xent but with explicit positive/negative pair handling.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negatives: Negative embeddings (batch_size, num_negatives, embedding_dim)
            
        Returns:
            InfoNCE loss value
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=2)
        
        # Compute positive similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # (batch_size,)
        
        # Compute negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / self.temperature  # (batch_size, num_neg)
        
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + num_neg)
        
        # Labels: positive is always at index 0
        labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class ContrastivePairSampler:
    """
    Sampler for creating positive and negative pairs for contrastive learning.
    """
    
    def __init__(
        self,
        num_negatives: int = 5,
        strategy: str = 'mixed',  # 'ticker', 'date', or 'mixed'
    ):
        """
        Initialize pair sampler.
        
        Args:
            num_negatives: Number of negative samples per positive
            strategy: Sampling strategy for negatives
                - 'ticker': Different ticker, same date
                - 'date': Same ticker, different date
                - 'mixed': Mix of both
        """
        self.num_negatives = num_negatives
        self.strategy = strategy
    
    def sample_negatives(
        self,
        batch_data: dict,
        positive_idx: int,
    ) -> list:
        """
        Sample negative pairs for a given positive sample.
        
        Args:
            batch_data: Dictionary with 'ticker', 'date', etc.
            positive_idx: Index of the positive sample
            
        Returns:
            List of negative sample indices
        """
        batch_size = len(batch_data['ticker'])
        positive_ticker = batch_data['ticker'][positive_idx]
        positive_date = batch_data['date'][positive_idx]
        
        negative_indices = []
        
        # Create candidate pool
        candidates = list(range(batch_size))
        candidates.remove(positive_idx)  # Exclude the positive sample itself
        
        if self.strategy == 'ticker':
            # Sample from different tickers, same date
            candidates = [
                i for i in candidates
                if batch_data['ticker'][i] != positive_ticker
                and batch_data['date'][i] == positive_date
            ]
        elif self.strategy == 'date':
            # Sample from same ticker, different date
            candidates = [
                i for i in candidates
                if batch_data['ticker'][i] == positive_ticker
                and batch_data['date'][i] != positive_date
            ]
        elif self.strategy == 'mixed':
            # Mix of both: different ticker OR different date
            candidates = [
                i for i in candidates
                if batch_data['ticker'][i] != positive_ticker
                or batch_data['date'][i] != positive_date
            ]
        
        # Sample negatives
        if len(candidates) >= self.num_negatives:
            import random
            negative_indices = random.sample(candidates, self.num_negatives)
        else:
            # If not enough candidates, sample with replacement or use all
            negative_indices = candidates
            if len(candidates) < self.num_negatives:
                # Pad with random samples from entire batch (excluding positive)
                all_candidates = [i for i in range(batch_size) if i != positive_idx]
                import random
                additional = random.sample(all_candidates, self.num_negatives - len(candidates))
                negative_indices.extend(additional)
        
        return negative_indices

