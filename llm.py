#!/usr/bin/env python3

import argparse
import os
import json
from dotenv import load_dotenv
from transcript_retriever import TranscriptRetriever

# For Groq API
try:
    from groq import Groq
except ImportError:
    print("Error: Groq Python SDK not installed.")
    print("To install, run: pip install groq")

# Load environment variables
load_dotenv()

def answer_with_groq(prompt, model="llama-3.3-70b-versatile", api_key=None):
    """
    Get an answer using Groq API with Llama model.
    
    Args:
        prompt: The prompt to send to the model
        model: The Groq model to use (default: llama-3.3-70b-versatile)
        api_key: Optional API key (defaults to environment variable)
        
    Returns:
        Generated answer as a string
    """
    # Get API key from environment if not provided
    api_key = api_key or os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("Groq API key is required. Set GROQ_API_KEY in your .env file or pass it as a parameter.")
    
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
    
    # Call Groq API
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.3,
        max_tokens=1024,
        stream=False,
    )
    
    # Extract and return the generated answer
    return chat_completion.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description='Answer questions about meetings using retrieval and Llama via Groq API')
    parser.add_argument('query', type=str, help='Question about the meeting')
    parser.add_argument('--meeting-id', '-m', type=str, help='Filter by specific meeting ID')
    parser.add_argument('--speakers', '-s', type=str, nargs='+', help='Filter by specific speakers')
    parser.add_argument('--limit', '-l', type=int, default=3, help='Maximum number of context chunks')
    parser.add_argument('--threshold', '-t', type=float, default=0.6, 
                        help='Minimum relevance score threshold (0-1)')
    parser.add_argument('--model', type=str, default='llama-3.3-70b-versatile',
                        help='Groq model to use (default: llama-3.3-70b-versatile)')
    parser.add_argument('--collection', '-c', type=str, 
                        help='Qdrant collection name (defaults to .env value)')
    parser.add_argument('--output-prompt', '-p', action='store_true',
                        help='Output the prompt sent to the LLM')
    parser.add_argument('--no-llm', '-n', action='store_true',
                        help='Skip LLM call and just output the prompt (for testing)')
    
    args = parser.parse_args()
    
    # Initialize the retriever
    retriever = TranscriptRetriever(collection_name=args.collection)
    
    print(f"Question: {args.query}")
    
    # Retrieve relevant chunks
    results = retriever.retrieve(
        query=args.query,
        limit=args.limit,
        threshold=args.threshold,
        meeting_id=args.meeting_id,
        speakers=args.speakers
    )
    
    if not results:
        print("No relevant context found for this question.")
        return
    
    print(f"Found {len(results)} relevant context chunks.")
    
    # Generate prompt for Llama
    prompt = retriever.generate_llm_prompt(args.query, results)
    
    # If no-llm is True or output-prompt is True, output the prompt
    if args.no_llm or args.output_prompt:
        print("\n--- Generated Prompt for LLM ---")
        print(prompt)
        if args.no_llm:
            return
    
    # Get answer using Groq API
    try:
        print(f"Querying Groq API with model: {args.model}...")
        answer = answer_with_groq(prompt, model=args.model)
        
        print("\n--- Answer ---")
        print(answer)
    
    except Exception as e:
        print(f"Error getting answer from Groq API: {e}")

if __name__ == "__main__":
    main()