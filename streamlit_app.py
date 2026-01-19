"""
Streamlit UI for Multilingual Text Processing
Modern web interface for the multilingual text processing system
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import our main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from multilingual_processor import MultilingualTextProcessor, ProcessingResult
except ImportError:
    st.error("Could not import the multilingual text processor. Please ensure multilingual_processor.py is in the same directory.")
    st.stop()

def create_streamlit_ui():
    """Create modern Streamlit UI for the multilingual text processor"""
    
    st.set_page_config(
        page_title="Multilingual Text Processing",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌍 Advanced Multilingual Text Processing")
    st.markdown("Process text in multiple languages using state-of-the-art NLP models")
    
    # Initialize processor
    if 'processor' not in st.session_state:
        with st.spinner("Initializing multilingual text processor..."):
            try:
                st.session_state.processor = MultilingualTextProcessor()
                st.success("✅ Processor initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing processor: {str(e)}")
                st.stop()
    
    processor = st.session_state.processor
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Task selection
    task = st.sidebar.selectbox(
        "Select Task",
        ["Text Classification", "Sentiment Analysis", "Named Entity Recognition"]
    )
    
    # Device info
    st.sidebar.markdown("---")
    st.sidebar.subheader("🖥️ System Info")
    st.sidebar.write(f"**Device:** {processor.device}")
    st.sidebar.write(f"**Available Languages:** {len(processor.get_available_languages())}")
    
    # Database stats
    stats = processor.get_database_stats()
    st.sidebar.write(f"**Text Samples:** {stats['total_samples']}")
    st.sidebar.write(f"**Processing Results:** {stats['total_results']}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"📝 {task}")
        
        # Text input
        text_input = st.text_area(
            "Enter text to process:",
            height=150,
            placeholder="Enter text in any supported language...",
            help="The system can process text in multiple languages including English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, and Korean."
        )
        
        # Task-specific parameters
        if task == "Text Classification":
            candidate_labels = st.text_input(
                "Candidate labels (comma-separated):",
                value="economy, sports, technology, politics, weather",
                help="Enter labels separated by commas"
            )
            labels = [label.strip() for label in candidate_labels.split(",") if label.strip()]
            
        elif task == "Named Entity Recognition":
            st.info("ℹ️ This will extract named entities like persons, organizations, locations, etc.")
        
        # Process button
        if st.button("🚀 Process Text", type="primary"):
            if text_input.strip():
                try:
                    with st.spinner("Processing..."):
                        if task == "Text Classification":
                            if not labels:
                                st.warning("Please provide at least one candidate label.")
                            else:
                                result = processor.classify_text(text_input, labels)
                        elif task == "Sentiment Analysis":
                            result = processor.analyze_sentiment(text_input)
                        elif task == "Named Entity Recognition":
                            result = processor.extract_entities(text_input)
                        
                        st.session_state.last_result = result
                        st.success("✅ Text processed successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Error processing text: {str(e)}")
            else:
                st.warning("⚠️ Please enter some text to process.")
    
    with col2:
        st.header("📊 Results")
        
        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            
            st.subheader("Latest Result")
            st.write(f"**Task:** {result.task.title()}")
            st.write(f"**Language:** {result.language}")
            st.write(f"**Confidence:** {result.confidence:.3f}")
            st.write(f"**Timestamp:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.subheader("Output")
            if result.task == "classification":
                st.write(f"**Label:** {result.result['labels'][0]}")
                st.write(f"**Score:** {result.result['scores'][0]:.3f}")
                
                # Show all scores
                df_scores = pd.DataFrame({
                    'Label': result.result['labels'],
                    'Score': result.result['scores']
                })
                st.bar_chart(df_scores.set_index('Label'))
                
            elif result.task == "sentiment":
                sentiment_label = result.result[0]['label']
                sentiment_score = result.result[0]['score']
                
                # Color coding for sentiment
                if sentiment_label == "POSITIVE":
                    st.success(f"**Sentiment:** {sentiment_label}")
                elif sentiment_label == "NEGATIVE":
                    st.error(f"**Sentiment:** {sentiment_label}")
                else:
                    st.info(f"**Sentiment:** {sentiment_label}")
                
                st.write(f"**Score:** {sentiment_score:.3f}")
                
                # Progress bar for sentiment score
                st.progress(sentiment_score)
                
            elif result.task == "ner":
                if result.result:
                    entities_df = pd.DataFrame(result.result)
                    st.dataframe(entities_df[['entity', 'word', 'score']])
                    
                    # Show entity counts
                    entity_counts = entities_df['entity'].value_counts()
                    if not entity_counts.empty:
                        st.subheader("Entity Distribution")
                        st.bar_chart(entity_counts)
                else:
                    st.write("No entities found")
        else:
            st.info("👆 Process some text to see results here")
    
    # Database section
    st.header("🗄️ Database")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Show Statistics"):
            stats = processor.get_database_stats()
            st.json(stats)
    
    with col2:
        if st.button("📝 Show Text Samples"):
            samples = processor.db.get_text_samples()
            if samples:
                df_samples = pd.DataFrame(samples)
                st.dataframe(df_samples[['text', 'language', 'category']])
            else:
                st.write("No text samples found")
    
    with col3:
        if st.button("🔍 Show Processing Results"):
            results = processor.db.get_processing_results()
            if results:
                df_results = pd.DataFrame(results)
                st.dataframe(df_results[['text', 'task', 'confidence', 'timestamp']])
            else:
                st.write("No processing results found")
    
    # Sample texts section
    st.header("📚 Sample Texts")
    
    # Show sample texts by language
    languages = processor.get_available_languages()
    selected_lang = st.selectbox("Select language to view samples:", languages)
    
    samples = processor.db.get_text_samples(selected_lang)
    if samples:
        st.write(f"**Sample texts in {selected_lang.upper()}:**")
        for sample in samples[:5]:  # Show first 5 samples
            st.write(f"• {sample['text']} ({sample['category']})")
    else:
        st.write(f"No samples found for {selected_lang}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, Transformers, and PyTorch")
    st.markdown("**Supported Languages:** English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean")

if __name__ == "__main__":
    create_streamlit_ui()
