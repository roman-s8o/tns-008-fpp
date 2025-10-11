# Quick Start Guide

Get up and running with the SSL-Based Financial Prediction Platform in minutes.

## Prerequisites

- Mac M3 (or M1/M2) with 16GB RAM
- Python 3.12.2
- Git

## Setup (5 minutes)

### 1. Create Virtual Environment

```bash
# Using venv (recommended)
python3.12 -m venv venv
source venv/bin/activate

# OR using conda
conda create -n fpp python=3.12.2
conda activate fpp
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Install spaCy model for NER
python -m spacy download en_core_web_sm
```

**Note**: Installation may take 5-10 minutes depending on your internet connection.

### 3. Verify Installation

```bash
# Run verification script
python scripts/verify_setup.py

# OR run tests
pytest tests/test_setup.py -v
```

Expected output:
```
✅ All checks passed! Environment is ready for development.
```

### 4. Configure Environment (Optional)

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# - Alpha Vantage: https://www.alphavantage.co/support/#api-key
# - Investing.com: (Get API access)
```

## Verify Mac M3 GPU Acceleration

```bash
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}'); print(f'MPS Built: {torch.backends.mps.is_built()}')"
```

Expected output:
```
MPS Available: True
MPS Built: True
```

## Common Issues

### Issue: PyTorch not using MPS

**Solution**: Ensure macOS >= 12.3 and PyTorch >= 2.1.0

```bash
pip install --upgrade torch torchvision torchaudio
```

### Issue: TA-Lib installation fails

**Solution**: TA-Lib is optional. Install via Homebrew if needed:

```bash
brew install ta-lib
pip install TA-Lib
```

Or use the `ta` library instead (already in requirements.txt).

### Issue: Memory errors during model loading

**Solution**: 
1. Enable 4-bit quantization in `config/config.yaml`
2. Reduce batch size
3. Use gradient accumulation

## Next Steps

### Milestone 2: Data Ingestion

Create your first data ingestion script:

```bash
# Create price ingestion script
touch src/data_ingestion/fetch_prices.py
```

Follow the ROADMAP.md for detailed milestone instructions.

### Quick Commands

```bash
# Run API (after Milestone 23)
uvicorn src.api.main:app --reload

# Run dashboard (after Milestone 24)
streamlit run src/dashboard/app.py

# Train model (after Milestone 9+)
python src/training/train.py

# Run tests
pytest tests/ -v --cov=src
```

## Project Structure

```
tns-008-fpp/
├── src/              # Source code
├── data/             # Data storage (gitignored)
├── scripts/          # Utility scripts
├── tests/            # Test suite
├── notebooks/        # Jupyter notebooks
├── config/           # Configuration files
└── logs/             # Logs (gitignored)
```

## Resources

- **ROADMAP.md**: Complete 28-milestone development plan
- **README.md**: Detailed project documentation
- **config/config.yaml**: Configuration settings

## Support

For issues or questions:
1. Check ROADMAP.md for milestone-specific guidance
2. Review logs in `logs/` directory
3. Run `pytest` to identify issues

Happy coding! 🚀

