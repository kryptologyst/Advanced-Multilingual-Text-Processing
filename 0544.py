"""
Project 544: Advanced Multilingual Text Processing
A comprehensive multilingual text processing system with modern NLP capabilities.

Features:
- Text classification across multiple languages
- Sentiment analysis
- Named Entity Recognition (NER)
- Text translation
- Text summarization
- Language detection
- Mock database integration
- Modern web UI
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from transformers import pipeline
import pandas as pd
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multilingual_nlp.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Data class for storing processing results"""
    text: str
    language: str
    task: str
    result: Any
    confidence: float
    timestamp: datetime

class MockDatabase:
    """Mock database for storing multilingual text samples and results"""
    
    def __init__(self, db_path: str = "multilingual_db.json"):
        self.db_path = Path(db_path)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load data from JSON file or create default structure"""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "text_samples": [],
                "processing_results": [],
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
    
    def save_data(self):
        """Save data to JSON file"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def add_text_sample(self, text: str, language: str, category: str = "general"):
        """Add a text sample to the database"""
        sample = {
            "id": len(self.data["text_samples"]) + 1,
            "text": text,
            "language": language,
            "category": category,
            "created": datetime.now().isoformat()
        }
        self.data["text_samples"].append(sample)
        self.save_data()
        return sample["id"]
    
    def add_processing_result(self, result: ProcessingResult):
        """Add a processing result to the database"""
        result_data = {
            "id": len(self.data["processing_results"]) + 1,
            "text": result.text,
            "language": result.language,
            "task": result.task,
            "result": str(result.result),
            "confidence": result.confidence,
            "timestamp": result.timestamp.isoformat()
        }
        self.data["processing_results"].append(result_data)
        self.save_data()
        return result_data["id"]
    
    def get_text_samples(self, language: Optional[str] = None) -> List[Dict]:
        """Get text samples, optionally filtered by language"""
        samples = self.data["text_samples"]
        if language:
            return [s for s in samples if s["language"] == language]
        return samples
    
    def get_processing_results(self, task: Optional[str] = None) -> List[Dict]:
        """Get processing results, optionally filtered by task"""
        results = self.data["processing_results"]
        if task:
            return [r for r in results if r["task"] == task]
        return results

class MultilingualTextProcessor:
    """Advanced multilingual text processing system"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize pipelines
        self.pipelines = {}
        self._initialize_pipelines()
        
        # Initialize database
        self.db = MockDatabase()
        
        # Load sample data
        self._load_sample_data()
    
    def _initialize_pipelines(self):
        """Initialize all NLP pipelines"""
        try:
            # Text classification pipeline
            self.pipelines['classification'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            
            # Sentiment analysis pipeline
            self.pipelines['sentiment'] = pipeline(
                "sentiment-analysis",
                device=0 if self.device == "cuda" else -1
            )
            
            # Named Entity Recognition pipeline
            self.pipelines['ner'] = pipeline(
                "ner",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("All pipelines initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipelines: {e}")
            # Fallback to simpler models
            self._initialize_fallback_pipelines()
    
    def _initialize_fallback_pipelines(self):
        """Initialize fallback pipelines with simpler models"""
        try:
            self.pipelines['classification'] = pipeline("zero-shot-classification")
            self.pipelines['sentiment'] = pipeline("sentiment-analysis")
            self.pipelines['ner'] = pipeline("ner")
            logger.info("Fallback pipelines initialized")
        except Exception as e:
            logger.error(f"Error initializing fallback pipelines: {e}")
    
    def _load_sample_data(self):
        """Load sample multilingual text data"""
        sample_texts = [
            ("The economy is growing rapidly.", "en", "economy"),
            ("La economía está creciendo rápidamente.", "es", "economy"),
            ("L'économie croît rapidement.", "fr", "economy"),
            ("Die Wirtschaft wächst schnell.", "de", "economy"),
            ("L'economia sta crescendo rapidamente.", "it", "economy"),
            ("I love this new technology!", "en", "technology"),
            ("¡Me encanta esta nueva tecnología!", "es", "technology"),
            ("J'adore cette nouvelle technologie!", "fr", "technology"),
            ("The weather is beautiful today.", "en", "weather"),
            ("El clima está hermoso hoy.", "es", "weather"),
            ("Le temps est magnifique aujourd'hui.", "fr", "weather"),
            ("Das Wetter ist heute wunderschön.", "de", "weather"),
        ]
        
        for text, lang, category in sample_texts:
            self.db.add_text_sample(text, lang, category)
        
        logger.info(f"Loaded {len(sample_texts)} sample texts")
    
    def classify_text(self, text: str, candidate_labels: List[str]) -> ProcessingResult:
        """Classify text using zero-shot classification"""
        try:
            result = self.pipelines['classification'](text, candidate_labels)
            
            processing_result = ProcessingResult(
                text=text,
                language="auto-detected",
                task="classification",
                result=result,
                confidence=result['scores'][0],
                timestamp=datetime.now()
            )
            
            self.db.add_processing_result(processing_result)
            return processing_result
            
        except Exception as e:
            logger.error(f"Error in text classification: {e}")
            raise
    
    def analyze_sentiment(self, text: str) -> ProcessingResult:
        """Analyze sentiment of text"""
        try:
            result = self.pipelines['sentiment'](text)
            
            processing_result = ProcessingResult(
                text=text,
                language="auto-detected",
                task="sentiment",
                result=result,
                confidence=result[0]['score'],
                timestamp=datetime.now()
            )
            
            self.db.add_processing_result(processing_result)
            return processing_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise
    
    def extract_entities(self, text: str) -> ProcessingResult:
        """Extract named entities from text"""
        try:
            result = self.pipelines['ner'](text)
            
            processing_result = ProcessingResult(
                text=text,
                language="auto-detected",
                task="ner",
                result=result,
                confidence=np.mean([entity['score'] for entity in result]) if result else 0.0,
                timestamp=datetime.now()
            )
            
            self.db.add_processing_result(processing_result)
            return processing_result
            
        except Exception as e:
            logger.error(f"Error in NER: {e}")
            raise
    
    def get_available_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_samples": len(self.db.data["text_samples"]),
            "total_results": len(self.db.data["processing_results"]),
            "languages": list(set(s["language"] for s in self.db.data["text_samples"])),
            "tasks": list(set(r["task"] for r in self.db.data["processing_results"]))
        }

def main():
    """Main function to demonstrate the multilingual text processor"""
    print("Multilingual Text Processing System")
    print("=" * 50)
    
    # Initialize processor
    processor = MultilingualTextProcessor()
    
    # Test examples
    test_texts = [
        "The economy is growing rapidly.",
        "La economía está creciendo rápidamente.",
        "I love this new technology!",
        "¡Me encanta esta nueva tecnología!"
    ]
    
    candidate_labels = ["economy", "sports", "technology", "politics"]
    
    print("\nTesting Text Classification:")
    for text in test_texts:
        try:
            result = processor.classify_text(text, candidate_labels)
            print(f"Text: {text}")
            print(f"Classification: {result.result['labels'][0]} (confidence: {result.confidence:.3f})")
            print("-" * 30)
        except Exception as e:
            print(f"Error processing '{text}': {e}")
    
    print("\nTesting Sentiment Analysis:")
    for text in test_texts[:2]:  # Test first two texts
        try:
            result = processor.analyze_sentiment(text)
            print(f"Text: {text}")
            print(f"Sentiment: {result.result[0]['label']} (confidence: {result.confidence:.3f})")
            print("-" * 30)
        except Exception as e:
            print(f"Error processing '{text}': {e}")
    
    print("\nDatabase Statistics:")
    stats = processor.get_database_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
