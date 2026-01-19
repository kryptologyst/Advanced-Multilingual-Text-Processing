#!/usr/bin/env python3
"""
Simple test script for Multilingual Text Processing
"""

def test_basic_functionality():
    """Test basic functionality without heavy dependencies"""
    print("🧪 Testing Multilingual Text Processing")
    print("=" * 50)
    
    try:
        # Test imports
        print("✅ Testing imports...")
        from multilingual_processor import MultilingualTextProcessor, MockDatabase, ProcessingResult
        from config import Config
        print("✅ All imports successful")
        
        # Test database
        print("\n✅ Testing database...")
        db = MockDatabase("test_db.json")
        sample_id = db.add_text_sample("Test text", "en", "test")
        print(f"✅ Added sample with ID: {sample_id}")
        
        # Test configuration
        print("\n✅ Testing configuration...")
        config = Config.get_model_config("classification")
        print(f"✅ Configuration loaded: {config['model_name']}")
        
        # Test ProcessingResult
        print("\n✅ Testing ProcessingResult...")
        from datetime import datetime
        result = ProcessingResult(
            text="Test text",
            language="en", 
            task="test",
            result={"test": "value"},
            confidence=0.95,
            timestamp=datetime.now()
        )
        print(f"✅ ProcessingResult created: {result.task}")
        
        print("\n🎉 All basic tests passed!")
        print("\nTo run the full application:")
        print("  Web interface: streamlit run streamlit_app.py")
        print("  Command line: python3 multilingual_processor.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if not success:
        exit(1)
