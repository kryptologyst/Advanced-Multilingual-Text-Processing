"""
Test suite for Multilingual Text Processing
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multilingual_processor import MultilingualTextProcessor, MockDatabase, ProcessingResult
from config import Config

class TestMockDatabase:
    """Test cases for MockDatabase class"""
    
    def setup_method(self):
        """Set up test database"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_file.close()
        self.db = MockDatabase(self.temp_file.name)
    
    def teardown_method(self):
        """Clean up test database"""
        os.unlink(self.temp_file.name)
    
    def test_database_initialization(self):
        """Test database initialization"""
        assert self.db.data is not None
        assert "text_samples" in self.db.data
        assert "processing_results" in self.db.data
        assert "metadata" in self.db.data
    
    def test_add_text_sample(self):
        """Test adding text samples"""
        sample_id = self.db.add_text_sample("Test text", "en", "test")
        assert sample_id == 1
        assert len(self.db.data["text_samples"]) == 1
        
        sample = self.db.data["text_samples"][0]
        assert sample["text"] == "Test text"
        assert sample["language"] == "en"
        assert sample["category"] == "test"
    
    def test_add_processing_result(self):
        """Test adding processing results"""
        result = ProcessingResult(
            text="Test text",
            language="en",
            task="classification",
            result={"label": "test"},
            confidence=0.95,
            timestamp=datetime.now()
        )
        
        result_id = self.db.add_processing_result(result)
        assert result_id == 1
        assert len(self.db.data["processing_results"]) == 1
    
    def test_get_text_samples(self):
        """Test retrieving text samples"""
        # Add samples
        self.db.add_text_sample("English text", "en", "test")
        self.db.add_text_sample("Spanish text", "es", "test")
        
        # Get all samples
        all_samples = self.db.get_text_samples()
        assert len(all_samples) == 2
        
        # Get filtered samples
        en_samples = self.db.get_text_samples("en")
        assert len(en_samples) == 1
        assert en_samples[0]["language"] == "en"
    
    def test_get_processing_results(self):
        """Test retrieving processing results"""
        # Add results
        result1 = ProcessingResult("Text 1", "en", "classification", {}, 0.9, datetime.now())
        result2 = ProcessingResult("Text 2", "es", "sentiment", {}, 0.8, datetime.now())
        
        self.db.add_processing_result(result1)
        self.db.add_processing_result(result2)
        
        # Get all results
        all_results = self.db.get_processing_results()
        assert len(all_results) == 2
        
        # Get filtered results
        classification_results = self.db.get_processing_results("classification")
        assert len(classification_results) == 1
        assert classification_results[0]["task"] == "classification"

class TestProcessingResult:
    """Test cases for ProcessingResult dataclass"""
    
    def test_processing_result_creation(self):
        """Test ProcessingResult creation"""
        timestamp = datetime.now()
        result = ProcessingResult(
            text="Test text",
            language="en",
            task="classification",
            result={"label": "test"},
            confidence=0.95,
            timestamp=timestamp
        )
        
        assert result.text == "Test text"
        assert result.language == "en"
        assert result.task == "classification"
        assert result.confidence == 0.95
        assert result.timestamp == timestamp

class TestMultilingualTextProcessor:
    """Test cases for MultilingualTextProcessor class"""
    
    def setup_method(self):
        """Set up test processor"""
        # Mock the transformers pipeline to avoid downloading models during tests
        with patch('multilingual_processor.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            self.processor = MultilingualTextProcessor()
    
    def test_processor_initialization(self):
        """Test processor initialization"""
        assert self.processor.device in ["cuda", "cpu"]
        assert hasattr(self.processor, 'pipelines')
        assert hasattr(self.processor, 'db')
    
    def test_get_available_languages(self):
        """Test getting available languages"""
        languages = self.processor.get_available_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
    
    def test_get_database_stats(self):
        """Test getting database statistics"""
        stats = self.processor.get_database_stats()
        assert isinstance(stats, dict)
        assert "total_samples" in stats
        assert "total_results" in stats
        assert "languages" in stats
        assert "tasks" in stats
    
    @patch('multilingual_processor.pipeline')
    def test_classify_text(self, mock_pipeline):
        """Test text classification"""
        # Mock the pipeline response
        mock_pipeline.return_value.return_value = {
            'labels': ['technology', 'economy', 'sports'],
            'scores': [0.8, 0.15, 0.05]
        }
        
        result = self.processor.classify_text("Test text", ["technology", "economy"])
        assert isinstance(result, ProcessingResult)
        assert result.task == "classification"
        assert result.confidence == 0.8
    
    @patch('multilingual_processor.pipeline')
    def test_analyze_sentiment(self, mock_pipeline):
        """Test sentiment analysis"""
        # Mock the pipeline response
        mock_pipeline.return_value.return_value = [{'label': 'POSITIVE', 'score': 0.95}]
        
        result = self.processor.analyze_sentiment("I love this!")
        assert isinstance(result, ProcessingResult)
        assert result.task == "sentiment"
        assert result.confidence == 0.95
    
    @patch('multilingual_processor.pipeline')
    def test_extract_entities(self, mock_pipeline):
        """Test named entity recognition"""
        # Mock the pipeline response
        mock_pipeline.return_value.return_value = [
            {'entity': 'B-PER', 'word': 'John', 'score': 0.95},
            {'entity': 'B-ORG', 'word': 'Apple', 'score': 0.90}
        ]
        
        result = self.processor.extract_entities("John works at Apple")
        assert isinstance(result, ProcessingResult)
        assert result.task == "ner"
        assert len(result.result) == 2

class TestConfig:
    """Test cases for Config class"""
    
    def test_config_initialization(self):
        """Test configuration initialization"""
        assert Config.PROJECT_NAME == "Multilingual Text Processing"
        assert Config.VERSION == "1.0.0"
        assert isinstance(Config.SUPPORTED_LANGUAGES, list)
        assert len(Config.SUPPORTED_LANGUAGES) > 0
    
    def test_get_model_config(self):
        """Test getting model configuration"""
        config = Config.get_model_config("classification")
        assert isinstance(config, dict)
        assert "model_name" in config
        assert "device" in config
        assert "max_length" in config
    
    def test_validate_config(self):
        """Test configuration validation"""
        assert Config.validate_config() == True

class TestIntegration:
    """Integration tests"""
    
    def setup_method(self):
        """Set up integration test"""
        with patch('multilingual_processor.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            self.processor = MultilingualTextProcessor()
    
    def test_end_to_end_classification(self):
        """Test end-to-end text classification"""
        with patch('multilingual_processor.pipeline') as mock_pipeline:
            mock_pipeline.return_value.return_value = {
                'labels': ['technology', 'economy'],
                'scores': [0.8, 0.2]
            }
            
            result = self.processor.classify_text("AI is amazing", ["technology", "economy"])
            
            # Check result
            assert result.task == "classification"
            assert result.confidence == 0.8
            
            # Check database
            stats = self.processor.get_database_stats()
            assert stats["total_results"] >= 1
    
    def test_multiple_language_processing(self):
        """Test processing text in multiple languages"""
        test_texts = [
            ("Hello world", "en"),
            ("Hola mundo", "es"),
            ("Bonjour le monde", "fr")
        ]
        
        with patch('multilingual_processor.pipeline') as mock_pipeline:
            mock_pipeline.return_value.return_value = {
                'labels': ['general', 'specific'],
                'scores': [0.7, 0.3]
            }
            
            for text, lang in test_texts:
                result = self.processor.classify_text(text, ["general", "specific"])
                assert result.task == "classification"
                assert result.text == text

# Fixtures for pytest
@pytest.fixture
def temp_db():
    """Fixture for temporary database"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.close()
    db = MockDatabase(temp_file.name)
    yield db
    os.unlink(temp_file.name)

@pytest.fixture
def mock_processor():
    """Fixture for mock processor"""
    with patch('0544.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        processor = MultilingualTextProcessor()
        yield processor

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
