#!/usr/bin/env python3
"""
Setup HuggingFace Authentication

This script helps set up HuggingFace authentication for accessing
gated models like Gemma.

Usage:
    python scripts/setup_hf_auth.py
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_huggingface_cli():
    """Check if huggingface-cli is installed."""
    try:
        result = subprocess.run(
            ["huggingface-cli", "--version"],
            capture_output=True,
            text=True
        )
        logger.info(f"✓ huggingface-cli installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.error("✗ huggingface-cli not found")
        logger.error("Please install: pip install huggingface-hub")
        return False


def check_token_env():
    """Check if HUGGINGFACE_TOKEN is set in environment."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        logger.info("✓ HUGGINGFACE_TOKEN found in environment")
        # Don't print the token for security
        logger.info(f"  Token length: {len(token)} characters")
        return True
    else:
        logger.warning("⚠ HUGGINGFACE_TOKEN not set in environment")
        return False


def check_hf_cache():
    """Check HuggingFace cache directory."""
    hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    token_path = Path(hf_home) / "token"
    
    if token_path.exists():
        logger.info(f"✓ HuggingFace token file found: {token_path}")
        return True
    else:
        logger.warning(f"⚠ No token file found at: {token_path}")
        return False


def prompt_login():
    """Prompt user to log in using huggingface-cli."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("HuggingFace Authentication Required")
    logger.info("=" * 70)
    logger.info("")
    logger.info("To access Gemma and other gated models, you need to:")
    logger.info("")
    logger.info("1. Create a HuggingFace account at https://huggingface.co/")
    logger.info("")
    logger.info("2. Accept the Gemma license:")
    logger.info("   https://huggingface.co/google/gemma-3-270m")
    logger.info("")
    logger.info("3. Get your access token:")
    logger.info("   https://huggingface.co/settings/tokens")
    logger.info("")
    logger.info("4. Authenticate using ONE of these methods:")
    logger.info("")
    logger.info("   Method A - CLI Login (Recommended):")
    logger.info("   $ huggingface-cli login")
    logger.info("")
    logger.info("   Method B - Environment Variable:")
    logger.info("   $ export HUGGINGFACE_TOKEN='your_token_here'")
    logger.info("")
    logger.info("   Method C - Add to .env file:")
    logger.info("   HUGGINGFACE_TOKEN=your_token_here")
    logger.info("")
    logger.info("=" * 70)
    logger.info("")
    
    response = input("Would you like to run 'huggingface-cli login' now? [y/N]: ")
    
    if response.lower() in ['y', 'yes']:
        logger.info("")
        logger.info("Launching huggingface-cli login...")
        logger.info("")
        try:
            subprocess.run(["huggingface-cli", "login"], check=True)
            logger.info("")
            logger.info("✓ Login complete!")
            return True
        except subprocess.CalledProcessError:
            logger.error("✗ Login failed")
            return False
        except KeyboardInterrupt:
            logger.info("")
            logger.info("Login cancelled")
            return False
    else:
        logger.info("Skipping login. Please authenticate manually.")
        return False


def create_env_template():
    """Create .env template if it doesn't exist."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    template = """# HuggingFace Authentication
HUGGINGFACE_TOKEN=your_token_here

# API Keys (for data ingestion)
ALPHA_VANTAGE_API_KEY=your_key_here
INVESTING_COM_API_KEY=your_key_here

# Weights & Biases (optional)
WANDB_API_KEY=your_key_here

# GCP (for deployment - Milestone 26)
GCP_PROJECT_ID=your_project_id
GCP_CREDENTIALS_PATH=path/to/credentials.json
"""
    
    if not env_example_path.exists():
        env_example_path.write_text(template)
        logger.info(f"✓ Created .env.example template")
    
    if not env_path.exists():
        logger.info(f"⚠ .env file not found")
        response = input("Create .env file from template? [y/N]: ")
        if response.lower() in ['y', 'yes']:
            env_path.write_text(template)
            logger.info(f"✓ Created .env file")
            logger.info(f"  Please edit .env and add your HuggingFace token")
            return True
    
    return False


def main():
    """Main setup function."""
    logger.info("=" * 70)
    logger.info("HuggingFace Authentication Setup")
    logger.info("=" * 70)
    logger.info("")
    
    # Check installation
    if not check_huggingface_cli():
        logger.info("")
        logger.info("Installing huggingface-hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"])
        logger.info("")
    
    # Check authentication status
    token_in_env = check_token_env()
    token_in_cache = check_hf_cache()
    
    logger.info("")
    
    if token_in_env or token_in_cache:
        logger.info("=" * 70)
        logger.info("✓ HuggingFace authentication appears to be configured")
        logger.info("=" * 70)
        logger.info("")
        logger.info("You should be able to access Gemma models.")
        logger.info("If you encounter authentication errors, make sure you've")
        logger.info("accepted the model license at:")
        logger.info("https://huggingface.co/google/gemma-3-270m")
        logger.info("")
        return 0
    else:
        # Prompt for login
        if prompt_login():
            return 0
        else:
            # Create .env template
            create_env_template()
            logger.info("")
            logger.info("Please complete authentication and try again.")
            return 1


if __name__ == "__main__":
    sys.exit(main())

