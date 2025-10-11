#!/usr/bin/env python3
"""
Verification script for Milestone 1: Project Setup
Checks that all required dependencies are installed and configured correctly.
"""

import sys
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)"


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        pkg = package_name or module_name
        return True, f"✓ {pkg}"
    except ImportError as e:
        pkg = package_name or module_name
        return False, f"✗ {pkg}: {str(e)}"


def check_torch_mps() -> Tuple[bool, str]:
    """Check PyTorch MPS (Metal Performance Shaders) availability for Mac."""
    try:
        import torch
        
        version = torch.__version__
        mps_available = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        
        if mps_available and mps_built:
            return True, f"✓ PyTorch {version} with MPS support (GPU acceleration enabled)"
        elif mps_built:
            return True, f"⚠ PyTorch {version} with MPS built but not available (check macOS version)"
        else:
            return False, f"✗ PyTorch {version} without MPS support"
    except Exception as e:
        return False, f"✗ PyTorch MPS check failed: {str(e)}"


def check_transformers() -> Tuple[bool, str]:
    """Check Hugging Face Transformers."""
    try:
        import transformers
        version = transformers.__version__
        return True, f"✓ Transformers {version}"
    except Exception as e:
        return False, f"✗ Transformers: {str(e)}"


def check_spacy_model() -> Tuple[bool, str]:
    """Check if spaCy model is downloaded."""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True, "✓ spaCy model 'en_core_web_sm' installed"
    except OSError:
        return False, "✗ spaCy model 'en_core_web_sm' not found (run: python -m spacy download en_core_web_sm)"
    except Exception as e:
        return False, f"✗ spaCy model check failed: {str(e)}"


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("Milestone 1: Project Setup Verification")
    print("=" * 70)
    print()
    
    checks: List[Tuple[bool, str]] = []
    
    # Python version
    print("1. Python Environment")
    print("-" * 70)
    status, msg = check_python_version()
    checks.append((status, msg))
    print(msg)
    print()
    
    # Core ML/DL libraries
    print("2. Core ML/DL Frameworks")
    print("-" * 70)
    
    status, msg = check_torch_mps()
    checks.append((status, msg))
    print(msg)
    
    for module, package in [
        ("torchvision", "torchvision"),
        ("torchaudio", "torchaudio"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    print()
    
    # Hugging Face
    print("3. Hugging Face Ecosystem")
    print("-" * 70)
    status, msg = check_transformers()
    checks.append((status, msg))
    print(msg)
    
    for module, package in [
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    print()
    
    # API Framework
    print("4. API Framework")
    print("-" * 70)
    for module, package in [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    print()
    
    # Data Processing
    print("5. Data Processing Libraries")
    print("-" * 70)
    for module, package in [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    print()
    
    # Financial Data
    print("6. Financial Data Libraries")
    print("-" * 70)
    for module, package in [
        ("yfinance", "yfinance"),
        ("alpha_vantage", "alpha-vantage"),
        ("requests", "requests"),
        ("bs4", "beautifulsoup4"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    print()
    
    # Database
    print("7. Database")
    print("-" * 70)
    for module, package in [
        ("sqlalchemy", "SQLAlchemy"),
        ("aiosqlite", "aiosqlite"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    print()
    
    # NLP & Feature Extraction
    print("8. NLP & Feature Extraction")
    print("-" * 70)
    for module, package in [
        ("spacy", "spaCy"),
        ("sklearn", "scikit-learn"),
        ("gensim", "gensim"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    
    status, msg = check_spacy_model()
    checks.append((status, msg))
    print(msg)
    print()
    
    # Technical Analysis
    print("9. Technical Analysis")
    print("-" * 70)
    status, msg = check_import("ta", "ta")
    checks.append((status, msg))
    print(msg)
    
    # Note: TA-Lib might be difficult to install on Mac
    status, msg = check_import("talib", "TA-Lib")
    if not status:
        print(f"⚠ {msg} (optional - can use 'ta' library instead)")
    else:
        checks.append((status, msg))
        print(msg)
    print()
    
    # Utilities
    print("10. Utilities")
    print("-" * 70)
    for module, package in [
        ("dotenv", "python-dotenv"),
        ("tqdm", "tqdm"),
        ("loguru", "loguru"),
        ("yaml", "PyYAML"),
        ("hydra", "hydra-core"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    print()
    
    # Monitoring & Visualization
    print("11. Monitoring & Visualization")
    print("-" * 70)
    for module, package in [
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    print()
    
    # Optional tools
    print("12. Optional Tools")
    print("-" * 70)
    for module, package in [
        ("wandb", "wandb"),
        ("streamlit", "Streamlit"),
        ("shap", "SHAP"),
    ]:
        status, msg = check_import(module, package)
        print(msg if status else f"⚠ {package} not installed (optional)")
    print()
    
    # Testing
    print("13. Testing Framework")
    print("-" * 70)
    for module, package in [
        ("pytest", "pytest"),
        ("httpx", "httpx"),
    ]:
        status, msg = check_import(module, package)
        checks.append((status, msg))
        print(msg)
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for status, _ in checks if status)
    total = len(checks)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"Passed: {passed}/{total} ({success_rate:.1f}%)")
    print()
    
    if success_rate == 100:
        print("✅ All checks passed! Environment is ready for development.")
        print()
        print("Next steps:")
        print("  1. Set up .env file with API keys")
        print("  2. Proceed to Milestone 2: Data Ingestion for Prices")
        return 0
    elif success_rate >= 90:
        print("⚠️  Most checks passed. Review warnings above.")
        print("   Environment is mostly ready, but some optional packages are missing.")
        return 0
    else:
        print("❌ Some critical checks failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        print()
        print("Failed checks:")
        for status, msg in checks:
            if not status:
                print(f"  {msg}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

