# Quick Start Guide - Gemma SSL Training

This guide helps you quickly get started with Gemma-3-270m SSL pre-training.

## Prerequisites Checklist

- ✅ Mac M3 with 16GB RAM
- ✅ MPS available (verified in test)
- ✅ Python environment activated
- ⚠️ HuggingFace authentication (see below)
- ⚠️ Gemma license accepted (see below)

## Step 1: HuggingFace Setup (First Time Only)

### Accept Gemma License
Visit: https://huggingface.co/google/gemma-3-270m and click "Accept License"

### Authenticate (Choose ONE method)

**Option A - CLI Login (Recommended):**
```bash
huggingface-cli login
# Paste your token when prompted
```

**Option B - Environment Variable:**
```bash
export HUGGINGFACE_TOKEN='your_token_here'
```

**Option C - Use Setup Script:**
```bash
python scripts/setup_hf_auth.py
```

## Step 2: Test Model Loading

```bash
python scripts/test_gemma_model.py
```

**Expected Output:**
```
✓ ALL TESTS PASSED!
Model: google/gemma-3-270m
Device: MPS
Total parameters: 268,098,176
Model size: ~0.54 GB (fp16)
```

## Step 3: Start Training

### Quick Training (Default Settings)
```bash
python scripts/train_gemma_ssl.py
```

### Custom Training
```bash
python scripts/train_gemma_ssl.py \
    --epochs 2 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --output-dir data/models/gemma
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-name` | google/gemma-3-270m | HuggingFace model |
| `--dataset-path` | data/processed | Dataset directory |
| `--output-dir` | data/models/gemma | Checkpoint directory |
| `--epochs` | 2 | Number of epochs |
| `--batch-size` | 4 | Batch size per device |
| `--learning-rate` | 1e-5 | Learning rate |

## Using Gemma in Your Code

### Basic Usage

```python
from src.models.gemma import load_gemma_model, load_gemma_tokenizer, GemmaConfig

# Create config
config = GemmaConfig()

# Load model and tokenizer
tokenizer = load_gemma_tokenizer(config)
model, device = load_gemma_model(config)

# Generate text
text = "The stock market today showed"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### Training with Custom Config

```python
from src.training import train_gemma_mlm
from src.models.gemma import GemmaConfig

# Custom config
config = GemmaConfig(
    batch_size=8,
    learning_rate=5e-6,
    num_epochs=3,
)

# Train
metrics = train_gemma_mlm(
    dataset_path="data/processed",
    output_dir="data/models/gemma",
    batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    num_epochs=config.num_epochs,
)

print(f"Final loss: {metrics['train_loss']:.4f}")
```

## Project Structure

```
src/models/gemma/          # Gemma model configuration
├── config.py              # GemmaConfig dataclass
├── model_loader.py        # Loading functions
└── README.md              # Detailed documentation

src/training/              # Training infrastructure
├── mlm_trainer.py         # MLM trainer class
└── __init__.py

scripts/                   # Executable scripts
├── setup_hf_auth.py       # HF authentication
├── test_gemma_model.py    # Test model loading
└── train_gemma_ssl.py     # Start training

data/models/gemma/         # Model checkpoints
└── checkpoint-*/          # Saved checkpoints
```

## Current Dataset Status

- **Training samples**: 139
- **Validation samples**: 17
- **Fine-tuning samples**: 18
- **Total**: 174 sequences

The dataset will grow as more news is collected. Current infrastructure supports millions of sequences.

## Performance Expectations

On Mac M3 with 16GB RAM:

| Metric | Value |
|--------|-------|
| Model loading | ~10-15 seconds |
| Training speed | ~5-10 sequences/sec |
| Epoch duration | ~30-60 seconds |
| Memory usage | ~4-6 GB |
| Total training time | ~2-5 minutes (2 epochs) |

## Configuration Options

Edit `src/models/gemma/config.py` or override in code:

```python
from src.models.gemma import GemmaConfig

config = GemmaConfig(
    # Training
    batch_size=4,                    # Batch size
    gradient_accumulation_steps=2,   # Effective batch = 8
    learning_rate=1e-5,              # Learning rate
    num_epochs=2,                    # Epochs
    
    # Model
    max_seq_length=512,              # Max tokens
    gradient_checkpointing=True,     # Save memory
    
    # MLM
    mlm_probability=0.15,            # Mask 15% tokens
    
    # Device
    device="mps",                    # MPS, cuda, or cpu
    torch_dtype="float16",           # Precision
)
```

## Troubleshooting

### "You are trying to access a gated repo"
- Accept the license at https://huggingface.co/google/gemma-3-270m
- Run `huggingface-cli login`

### "MPS backend out of memory"
- Reduce `batch_size` to 2
- Close other applications
- Restart terminal/notebook

### "No module named 'src'"
Make sure you're running from the project root:
```bash
cd /Users/romanshevchenko/Code/tns-008-fpp
python scripts/test_gemma_model.py
```

### Training is slow
- First epoch is always slower (model compilation)
- Expected: ~5-10 sequences/second on M3
- Use `--batch-size 8` for faster training (if memory allows)

## Monitoring Training

Training logs are saved to:
- Console output (real-time)
- `data/models/gemma/logs/` (TensorBoard logs)
- `data/models/gemma/trainer_state.json` (training state)

### View with TensorBoard
```bash
pip install tensorboard
tensorboard --logdir data/models/gemma/logs
```

## Next Steps

After successful training:

1. ✅ **Milestone 6 Complete** - Models load successfully
2. ⏭ **Milestone 7** - Implement MLM with 15% token masking
3. ⏭ **Milestone 8** - Add contrastive learning
4. ⏭ **Milestone 9** - Pre-train on full dataset

## Files to Know

| File | Purpose |
|------|---------|
| `src/models/gemma/config.py` | Configuration |
| `src/models/gemma/model_loader.py` | Model loading |
| `src/training/mlm_trainer.py` | Training logic |
| `scripts/test_gemma_model.py` | Test model |
| `scripts/train_gemma_ssl.py` | Train model |
| `MILESTONE_6_SUMMARY.md` | Detailed summary |

## Support

For issues:
1. Check logs in console output
2. Read `src/models/gemma/README.md`
3. Review `MILESTONE_6_SUMMARY.md`
4. Check error traceback

## Success Indicators

✅ Model loads in ~10-15 seconds  
✅ Forward pass works  
✅ Training starts without errors  
✅ Loss decreases over time  
✅ Checkpoints save successfully  
✅ MPS shows in device info  

---

**Ready to start training!** Run `python scripts/train_gemma_ssl.py`

