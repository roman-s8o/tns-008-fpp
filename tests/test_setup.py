"""
Basic tests to verify project setup and dependencies.
"""

import sys
import pytest


def test_python_version():
    """Test that Python version is 3.10 or higher."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor >= 10, "Python 3.10+ required"


def test_import_torch():
    """Test PyTorch import."""
    import torch
    assert torch.__version__ is not None


def test_torch_mps_available():
    """Test MPS (Metal Performance Shaders) availability on Mac."""
    import torch
    # MPS should be built, but availability depends on macOS version
    assert torch.backends.mps.is_built(), "PyTorch should have MPS support built"


def test_import_transformers():
    """Test Hugging Face Transformers import."""
    import transformers
    assert transformers.__version__ is not None


def test_import_fastapi():
    """Test FastAPI import."""
    import fastapi
    assert fastapi.__version__ is not None


def test_import_pandas():
    """Test pandas import."""
    import pandas
    assert pandas.__version__ is not None


def test_import_numpy():
    """Test numpy import."""
    import numpy
    assert numpy.__version__ is not None


def test_project_structure(tmp_path):
    """Test that essential project directories exist."""
    import os
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    
    # Check essential directories
    essential_dirs = [
        "src",
        "src/data_ingestion",
        "src/preprocessing",
        "src/models",
        "src/training",
        "src/api",
        "data",
        "data/raw",
        "data/processed",
        "data/models",
        "scripts",
        "tests",
        "notebooks",
        "config",
        "logs",
    ]
    
    for dir_path in essential_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Directory {dir_path} should exist"


def test_requirements_file_exists():
    """Test that requirements.txt exists."""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    
    assert requirements_file.exists(), "requirements.txt should exist"
    
    # Check that it's not empty
    content = requirements_file.read_text()
    assert len(content) > 0, "requirements.txt should not be empty"
    assert "torch" in content, "requirements.txt should include torch"
    assert "transformers" in content, "requirements.txt should include transformers"


def test_gitignore_exists():
    """Test that .gitignore exists."""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    gitignore_file = project_root / ".gitignore"
    
    assert gitignore_file.exists(), ".gitignore should exist"
    
    # Check that it contains essential entries
    content = gitignore_file.read_text()
    assert "__pycache__" in content
    assert ".env" in content
    assert "*.db" in content


def test_readme_exists():
    """Test that README.md exists."""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    readme_file = project_root / "README.md"
    
    assert readme_file.exists(), "README.md should exist"
    
    content = readme_file.read_text()
    assert len(content) > 0, "README.md should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

