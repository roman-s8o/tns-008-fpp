# Milestone 17: FinBERT Fine-Tuning Results (EUR/USD)
**Date**: October 20, 2025  
**Model**: FinBERT Contrastive with Multi-Task Head  
**Task**: EUR/USD Direction & Bucket Prediction

---

## 📊 Executive Summary

**Training completed in 488.3 minutes (~8.1 hours) over 10 epochs.**

### Performance Overview
| Metric | Result | Target | Status |
|--------|---------|---------|---------|
| **Direction Accuracy** | **53.8%** | 70% | ❌ **FAILED** |
| **Bucket Accuracy** | 45.4% | N/A | ⚠️ Low |
| **Bucket MAE** | 0.696 | N/A | ⚠️ Moderate |
| **Validation Loss** | 1.0508 | N/A | - |

**Conclusion**: The model performs **barely better than random guessing (50%)** and falls significantly short of the 70% target.

---

## 📈 Training Progression

### Epoch-by-Epoch Results

| Epoch | Train Loss | Val Loss | Direction Acc | Bucket Acc | Bucket MAE | Notes |
|-------|-----------|----------|---------------|------------|------------|-------|
| 1 | 1.1469 | 1.0900 | 51.2% | 45.4% | 0.696 | Baseline |
| 2 | 1.1147 | 1.0653 | 52.3% | 45.4% | 0.696 | Improving |
| 3 | 1.1113 | 1.0549 | 53.1% | 45.4% | 0.696 | Improving |
| **4** | **1.1081** | **1.0508** | **53.8%** | **45.4%** | **0.696** | **BEST** ✅ |
| 5 | 1.1035 | 1.0499 | 51.2% | 45.4% | 0.696 | Declined |
| 6 | 1.1052 | 1.0493 | 51.9% | 45.4% | 0.696 | Unstable |
| 7 | 1.1046 | 1.0487 | 51.9% | 45.4% | 0.696 | Plateaued |
| 8 | 1.1043 | 1.0485 | 51.9% | 45.4% | 0.696 | Plateaued |
| 9 | 1.1041 | 1.0483 | 51.9% | 45.4% | 0.696 | Plateaued |
| 10 | 1.1047 | 1.0483 | 51.9% | 45.4% | 0.696 | Plateaued |

### Key Observations
1. ✅ **Best performance at Epoch 4** (53.8% direction accuracy)
2. ⚠️ **Performance degraded** after epoch 4, then plateaued
3. ⚠️ **Bucket predictions never improved** (stuck at 45.4% throughout)
4. ⚠️ **No improvement** for 6 consecutive epochs (5-10)
5. ⚠️ **Early stopping should have triggered** at epoch 7 (3 epochs without improvement)

---

## 🔍 Detailed Analysis

### 1. Direction Prediction Performance
- **Baseline (Random Guessing)**: 50.0%
- **Achieved**: 53.8%
- **Improvement over baseline**: +3.8 percentage points
- **Gap to target (70%)**: -16.2 percentage points

**Verdict**: The model shows minimal learning and barely outperforms random chance.

### 2. Bucket Prediction Performance
- **Expected Random Performance**: 20% (5 buckets)
- **Achieved**: 45.4%
- **Improvement over baseline**: +25.4 percentage points
- **MAE**: 0.696 (average error of ~0.7 buckets)

**Verdict**: Bucket predictions are better than random but still weak. The model may be biasing toward the middle bucket.

### 3. Training Dynamics
- **Loss decreased steadily** but performance plateaued
- **Overfitting unlikely**: Train loss and val loss moved together
- **Underfitting likely**: Model unable to capture forex patterns

---

## 🚨 Root Cause Analysis

### Why is performance so low?

#### 1. **Data Quality Issues**
- **Limited training data**: 2,080 training samples may be insufficient
- **Forex is highly stochastic**: EUR/USD movements are notoriously difficult to predict
- **News-price lag**: News may not align temporally with price movements
- **Missing key signals**: Technical indicators alone may not capture all market dynamics

#### 2. **Model Architecture Issues**
- **Base FinBERT**: Pre-trained on financial news but not forex-specific
- **Multi-task head**: May be too simple (single linear layer per task)
- **LoRA parameters**: r=16, alpha=32 may be too conservative

#### 3. **Training Configuration Issues**
- **Learning rate (1e-5)**: May be too low for effective learning
- **Batch size (8)**: Reduced from 16, may cause noisy gradients
- **Loss weighting (50/50)**: May not be optimal for this task

#### 4. **Feature Engineering Issues**
- **Text formatting**: Structured tags may not be ideal for BERT
- **Feature scaling**: Numerical features may not be properly normalized
- **Missing features**: Sentiment, NER, topics not explicitly included in final format

---

## 📋 Comparison to Baseline

| Approach | Direction Acc | Notes |
|----------|---------------|-------|
| **Random Guessing** | 50.0% | Coin flip |
| **Always Predict Up** | ~50-55% | Depends on market trend |
| **Technical Indicators Only** | 45-60% | Literature benchmark |
| **Our FinBERT Model** | **53.8%** | Below expectations |
| **Target** | **70%** | Industry standard for profitable trading |

**Verdict**: Our model is in the lower range of technical indicator baselines and far from profitable trading thresholds.

---

## 🔧 Recommended Actions

### Immediate Actions (Priority 1)

1. **Increase Learning Rate**
   - Current: 1e-5
   - Recommended: 2e-5 or 3e-5
   - Rationale: Faster convergence and better optimization

2. **Increase Batch Size**
   - Current: 8
   - Recommended: 16 or 32
   - Rationale: More stable gradients, better generalization

3. **Adjust LoRA Parameters**
   - Current: r=16, alpha=32
   - Recommended: r=32, alpha=64 or full fine-tuning
   - Rationale: Allow more model capacity to learn

4. **Extend Training**
   - Current: 10 epochs
   - Recommended: 30-50 epochs with early stopping
   - Rationale: Model may need more time to converge

### Medium-Term Actions (Priority 2)

5. **Improve Feature Engineering**
   - Include sentiment scores explicitly as numerical features
   - Add entity-based features (e.g., Fed mentioned, ECB mentioned)
   - Add topic distributions as features
   - Experiment with different text formatting (remove tags, use natural language)

6. **Enhance Model Architecture**
   - Add deeper multi-task heads (2-3 layers with dropout)
   - Experiment with attention pooling instead of CLS token
   - Add skip connections or residual blocks

7. **Data Augmentation**
   - Collect more forex news (currently limited)
   - Use sliding window for temporal sequences
   - Experiment with different time horizons (2-day, 3-day predictions)

### Long-Term Actions (Priority 3)

8. **Try Different Models**
   - Fine-tune Gemma 2B (already have SSL pre-training)
   - Try forex-specific transformers if available
   - Ensemble multiple models

9. **Improve Data Pipeline**
   - Add forex news from paid sources (Bloomberg, Reuters)
   - Include economic calendar features more explicitly
   - Add order flow data, sentiment indices

10. **Rethink the Problem**
    - Consider regression instead of classification
    - Try multi-step prediction (1-day, 3-day, 1-week)
    - Focus on volatility prediction instead of direction

---

## 🎯 Next Steps

Based on user instructions:
> "If we don't hit 70% direction accuracy, extend training and adjust hyperparameters"

### Proposed Training Run #2

**Adjustments:**
- Learning rate: **2e-5** (↑ from 1e-5)
- Batch size: **16** (↑ from 8)
- LoRA rank: **32** (↑ from 16)
- LoRA alpha: **64** (↑ from 32)
- Epochs: **30** with early stopping (patience=5)
- Loss weighting: **60% direction, 40% buckets** (prioritize direction)

**Expected improvements:**
- Faster convergence with higher LR
- More stable training with larger batch size
- More model capacity with larger LoRA parameters
- Better optimization with more epochs

**Estimated time:** ~15-20 hours

---

## 📊 Detailed Metrics

### Training Set Performance
- **Samples**: 2,080
- **Final Loss**: 1.1047
- **Direction Accuracy**: N/A (not tracked on training set during training)

### Validation Set Performance
- **Samples**: 260
- **Best Loss**: 1.0508 (epoch 4)
- **Best Direction Accuracy**: 53.8% (epoch 4)
- **Best Bucket Accuracy**: 45.4% (all epochs)
- **Best Bucket MAE**: 0.696 (all epochs)

### Confusion Matrix Analysis
*(Would require loading the model and running predictions)*

---

## 💾 Artifacts

- **Model checkpoint**: `data/models/finbert_forex_finetuned/best/`
- **Training history**: `data/models/finbert_forex_finetuned/training_history.json`
- **Training logs**: Console output (488.3 minutes)

---

## ⏱️ Performance Metrics

- **Training time**: 488.3 minutes (~8.1 hours)
- **Time per epoch**: ~48.8 minutes
- **Time per batch**: ~11.3 seconds
- **Throughput**: ~0.7 samples/second

---

## 🔚 Conclusion

The fine-tuning run completed successfully but **failed to meet the 70% direction accuracy target**. The model shows minimal learning (53.8% vs 50% random baseline) and requires significant improvements.

**Recommendation**: Proceed with Training Run #2 with adjusted hyperparameters before considering architectural changes or data improvements.

---

**Generated**: October 20, 2025  
**Status**: ⚠️ REQUIRES REMEDIATION

