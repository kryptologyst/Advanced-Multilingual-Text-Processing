"""
Configuration file for Multilingual Text Processing
"""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration class for the multilingual text processing system"""
    
    # Project settings
    PROJECT_NAME = "Multilingual Text Processing"
    VERSION = "1.0.0"
    AUTHOR = "AI Projects"
    
    # File paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    
    # Database settings
    DB_PATH = BASE_DIR / "multilingual_db.json"
    DB_BACKUP_PATH = BASE_DIR / "backups"
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "multilingual_nlp.log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Model settings
    DEFAULT_MODELS = {
        "classification": "facebook/bart-large-mnli",
        "sentiment": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "ner": "xlm-roberta-large-finetuned-conll03-english",
        "translation": "facebook/mbart-large-50-many-to-many-mmt",
        "summarization": "facebook/mbart-large-50-many-to-many-mmt"
    }
    
    FALLBACK_MODELS = {
        "classification": "facebook/bart-large-mnli",
        "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
        "ner": "dbmdz/bert-large-cased-finetuned-conll03-english"
    }
    
    # Supported languages
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"
    ]
    
    LANGUAGE_NAMES = {
        "en": "English",
        "es": "Spanish", 
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean"
    }
    
    # Processing settings
    MAX_TEXT_LENGTH = 512
    BATCH_SIZE = 32
    DEVICE = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # UI settings
    STREAMLIT_CONFIG = {
        "page_title": "Multilingual Text Processing",
        "page_icon": "🌍",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # Performance settings
    ENABLE_CACHE = True
    CACHE_SIZE = 1000
    TIMEOUT_SECONDS = 30
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.DB_BACKUP_PATH.mkdir(exist_ok=True)
    
    @classmethod
    def get_model_config(cls, task: str) -> Dict[str, Any]:
        """Get model configuration for a specific task"""
        return {
            "model_name": cls.DEFAULT_MODELS.get(task),
            "fallback_model": cls.FALLBACK_MODELS.get(task),
            "device": cls.DEVICE,
            "max_length": cls.MAX_TEXT_LENGTH,
            "batch_size": cls.BATCH_SIZE
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        try:
            # Check if required directories exist
            cls.create_directories()
            
            # Validate language codes
            for lang in cls.SUPPORTED_LANGUAGES:
                if lang not in cls.LANGUAGE_NAMES:
                    return False
            
            return True
        except Exception:
            return False

# Initialize configuration
Config.create_directories()
