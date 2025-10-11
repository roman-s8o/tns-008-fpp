# SSL-Based Financial Prediction Platform

An advanced machine learning platform for predicting Nasdaq-100, FOREX, and S&P Futures market movements using Self-Supervised Learning (SSL) on financial news and price data.

## 🎯 Project Overview

This platform ingests daily financial news (up to 2000 articles/day) and historical stock prices (5 years) to predict:
- **Direction**: Market movement (up/down)
- **% Change Buckets**: Magnitude of price changes

**Core Technologies**:
- Self-Supervised Learning with FinBERT, Phi-3-mini, and Gemma-3-4B
- Daily incremental learning with LoRA fine-tuning
- FastAPI backend with optional Streamlit dashboard
- Deployed on GCP

## 📁 Project Structure

```
tns-008-fpp/
├── src/                          # Source code
│   ├── data_ingestion/          # Data fetching scripts
│   ├── preprocessing/           # Data cleaning and normalization
│   ├── models/                  # Model architectures and configs
│   ├── training/                # Training and fine-tuning scripts
│   └── api/                     # FastAPI application
├── data/                        # Data storage
│   ├── raw/                     # Raw financial data
│   ├── processed/               # Preprocessed datasets
│   └── models/                  # Trained model checkpoints
├── scripts/                     # Utility and automation scripts
├── tests/                       # Unit and integration tests
├── notebooks/                   # Jupyter notebooks for experimentation
├── config/                      # Configuration files
├── logs/                        # Application and training logs
├── requirements.txt             # Python dependencies
└── ROADMAP.md                   # Development roadmap (28 milestones)
```

## 🚀 Setup Instructions

### Prerequisites

- **Python**: 3.12.2
- **Hardware**: Mac M3 (or M1/M2) with 16GB RAM
- **OS**: macOS

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd tns-008-fpp
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Using venv
   python3.12 -m venv venv
   source venv/bin/activate
   
   # OR using conda
   conda create -n fpp python=3.12.2
   conda activate fpp
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install spaCy language model** (for NER):
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   # API Keys
   ALPHA_VANTAGE_API_KEY=your_key_here
   INVESTING_COM_API_KEY=your_key_here
   
   # Weights & Biases (optional)
   WANDB_API_KEY=your_key_here
   
   # GCP (for deployment - Milestone 26)
   GCP_PROJECT_ID=your_project_id
   GCP_CREDENTIALS_PATH=path/to/credentials.json
   ```

6. **Verify installation**:
   ```bash
   python scripts/verify_setup.py
   ```

## 💻 Mac M3 Optimization

This project is optimized for Apple Silicon (M3/M2/M1):

- **PyTorch MPS Backend**: Leverages Metal Performance Shaders for GPU acceleration
- **Quantization**: 4-bit quantization for Gemma-3-4B to reduce memory footprint
- **LoRA Fine-tuning**: Efficient parameter updates for incremental learning
- **Batch Sizes**: Optimized for 16GB RAM

### Checking MPS Availability

```python
import torch
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"MPS Built: {torch.backends.mps.is_built()}")
```

## 📊 Data Sources

- **Stock Prices**: Yahoo Finance (yfinance)
- **News Articles**: 
  - Alpha Vantage API (up to 2000 articles/day)
  - Investing.com API
- **Markets**: 
  - Nasdaq-100 (MVP)
  - FOREX (Milestone 27)
  - S&P Futures (Milestone 27)

## 🧠 Models

1. **FinBERT**: Pre-trained BERT for financial text
2. **Phi-3-mini**: Lightweight Microsoft SLM
3. **Gemma-3-4B**: Google's 4B parameter model (4-bit quantized)

## 🛠️ Development Workflow

### Current Milestone: 1 - Project Setup ✅

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Code Formatting

```bash
black src/ scripts/ tests/
flake8 src/ scripts/ tests/
mypy src/
```

## 📈 Training Pipeline (Overview)

1. **Data Ingestion**: Fetch news + prices
2. **Preprocessing**: Clean, normalize, create sequences
3. **SSL Pre-training**: MLM + Contrastive Learning
4. **Feature Extraction**: Sentiment, NER, Topics, Technical Indicators
5. **Fine-tuning**: Classification (direction) + Regression (% change)
6. **Incremental Learning**: Daily LoRA updates
7. **Prediction**: FastAPI endpoint

## 🌐 API Usage (Milestone 23+)

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "news": ["Article 1 text...", "Article 2 text..."],
        "prices": {
            "open": 150.2,
            "high": 152.1,
            "low": 149.8,
            "close": 151.5,
            "volume": 1000000
        }
    }
)

print(response.json())
# Output: {"direction": "up", "change_bucket": "1-2%", "confidence": 0.85}
```

## 📦 Deployment (Milestone 26+)

- **Platform**: Google Cloud Platform (GCP)
- **API**: Cloud Run
- **Database**: BigQuery
- **Scheduling**: Cloud Scheduler (daily updates)
- **Monitoring**: Stackdriver

## 🎯 Success Metrics

- **Direction Accuracy**: >70%
- **% Change MAE**: <2%
- **API Latency**: <5 seconds
- **Throughput**: >100 predictions/min
- **Uptime**: 99.9%

## 🔧 Troubleshooting

### Common Issues

1. **PyTorch not using MPS**:
   - Ensure macOS >= 12.3
   - Update PyTorch: `pip install --upgrade torch torchvision torchaudio`

2. **Out of Memory Errors**:
   - Reduce batch size
   - Enable 4-bit quantization
   - Use gradient accumulation

3. **API Rate Limits**:
   - Implement exponential backoff
   - Use multiple API keys (if allowed)



