import streamlit as st
import os
import json
from dotenv import load_dotenv
import sys
import subprocess
from typing import List, Dict, Any, Optional
import time

# Import your retriever
try:
    from transcript_retriever import TranscriptRetriever
except ImportError:
    st.error("Could not import TranscriptRetriever. Make sure the file is in the same directory.")
    st.stop()

# Try to import Groq if available
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Meeting Context Q&A",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 18px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .context-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        max-height: 300px;
        overflow-y: auto;
    }
    .speaker-text {
        margin-bottom: 8px;
    }
    .speaker-name {
        font-weight: bold;
        color: #2C3E50;
    }
</style>
""", unsafe_allow_html=True)

def answer_with_groq(prompt, model="llama-3.3-70b-versatile", api_key=None):
    """
    Get an answer using Groq API with Llama model.
    
    Args:
        prompt: The prompt to send to the model
        model: The Groq model to use
        api_key: Optional API key (defaults to environment variable)
        
    Returns:
        Generated answer as a string
    """
    if not GROQ_AVAILABLE:
        return "Error: Groq Python SDK not installed. Please run: pip install groq"
    
    # Get API key from environment if not provided
    api_key = api_key or os.getenv("GROQ_API_KEY")
    
    if not api_key:
        return "Error: Groq API key is required. Please set GROQ_API_KEY in your .env file or enter it in the settings."
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Format messages for chat completion
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that answers questions about meetings based on transcript excerpts."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        # Call Groq API with progress indicator
        with st.spinner("Generating answer..."):
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.3,
                max_tokens=1024,
                stream=False,
            )
        
        # Extract and return the generated answer
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error accessing Groq API: {str(e)}"

def format_context(results):
    """Format the context for display in the UI"""
    if not results:
        return "No relevant context found."
    
    context_html = ""
    
    for i, result in enumerate(results):
        context_html += f"<h4>Relevant Transcript {i+1} (Score: {result['score']:.2f})</h4>"
        context_html += f"<p>Meeting: {result['title']}</p>"
        
        # Display segments with speaker information
        for segment in result.get('segments', []):
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '')
            context_html += f'<div class="speaker-text"><span class="speaker-name">{speaker}:</span> {text}</div>'
        
        context_html += "<hr>"
    
    return context_html

def main():
    # Sidebar for configuration
    st.sidebar.title("Settings")
    
    # API key input (with option to use from .env)
    use_env_api_key = st.sidebar.checkbox("Use API key from .env file", value=True)
    if use_env_api_key:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            st.sidebar.warning("Warning: GROQ_API_KEY not found in .env file")
    else:
        api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    # Model selection
    model = st.sidebar.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    )
    
    # Retrieval settings
    st.sidebar.subheader("Retrieval Settings")
    limit = st.sidebar.slider("Number of context chunks", min_value=1, max_value=10, value=3)
    threshold = st.sidebar.slider("Relevance threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    
    # Meeting filter options
    st.sidebar.subheader("Filters")
    
    # Dynamically get available meetings if possible
    try:
        retriever = TranscriptRetriever()
        # This is hypothetical - you'd need to add this method to your retriever
        # meetings = retriever.get_available_meetings()
        meetings = []  # Placeholder until you implement this
    except:
        meetings = []
    
    meeting_id = st.sidebar.selectbox(
        "Filter by Meeting",
        ["All Meetings"] + meetings
    )
    
    meeting_id = None if meeting_id == "All Meetings" else meeting_id
    
    # Show context toggle
    show_context = st.sidebar.checkbox("Show retrieved context", value=False)
    
    # Main UI
    st.title("Meeting Context Q&A")
    st.markdown("Ask questions about your meeting transcripts and get answers based on the stored content.")
    
    # Query input
    query = st.text_area("Enter your question about the meetings:", height=100)
    
    # Submit button
    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a question.")
        else:
            # Initialize retriever
            try:
                retriever = TranscriptRetriever()
                
                with st.spinner("Searching for relevant context..."):
                    # Retrieve relevant context
                    results = retriever.retrieve(
                        query=query,
                        limit=limit,
                        threshold=threshold,
                        meeting_id=meeting_id
                    )
                
                if not results:
                    st.warning("No relevant context found for your question.")
                else:
                    st.success(f"Found {len(results)} relevant segments from the meeting transcripts.")
                    
                    # Display context if requested
                    if show_context:
                        st.markdown("<div class='context-box'>", unsafe_allow_html=True)
                        st.markdown(format_context(results), unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Generate prompt for LLM
                    prompt = retriever.generate_llm_prompt(query, results)
                    
                    # Get answer using Groq
                    answer = answer_with_groq(prompt, model=model, api_key=api_key)
                    
                    # Display answer
                    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                    st.markdown(f"<h3>Answer:</h3>", unsafe_allow_html=True)
                    st.write(answer)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Information about the app
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app retrieves relevant context from your meeting transcripts and uses "
        "Groq's API to generate answers based on that context."
    )

if __name__ == "__main__":
    main()