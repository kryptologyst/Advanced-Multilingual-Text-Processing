#!/usr/bin/env python3
"""
Setup script for Multilingual Text Processing
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ["data", "logs", "models", "backups"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_gpu_support():
    """Check for GPU support"""
    print("🖥️ Checking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ CUDA not available, using CPU")
    except ImportError:
        print("⚠️ PyTorch not installed yet")

def run_tests():
    """Run test suite"""
    print("🧪 Running tests...")
    try:
        subprocess.check_call([sys.executable, "-m", "pytest", "test_multilingual.py", "-v"])
        print("✅ All tests passed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False
    except FileNotFoundError:
        print("⚠️ pytest not found, skipping tests")
    return True

def create_sample_data():
    """Create sample data files"""
    print("📊 Creating sample data...")
    
    # Create a sample configuration
    sample_config = {
        "project": "Multilingual Text Processing",
        "version": "1.0.0",
        "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
        "models": {
            "classification": "facebook/bart-large-mnli",
            "sentiment": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "ner": "xlm-roberta-large-finetuned-conll03-english"
        }
    }
    
    with open("data/sample_config.json", "w") as f:
        import json
        json.dump(sample_config, f, indent=2)
    
    print("✅ Sample data created")

def main():
    """Main setup function"""
    print("🌍 Multilingual Text Processing Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Check GPU support
    check_gpu_support()
    
    # Create sample data
    create_sample_data()
    
    # Run tests
    tests_passed = run_tests()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nTo run the application:")
    print("  Command line: python 0544.py")
    print("  Web interface: streamlit run streamlit_app.py")
    print("\nTo run with Docker:")
    print("  docker-compose up")
    
    if not tests_passed:
        print("\n⚠️ Some tests failed. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
