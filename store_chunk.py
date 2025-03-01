#!/usr/bin/env python3

import json
import os
import argparse
import uuid
from datetime import datetime

# For environment variables
from dotenv import load_dotenv

# Import the transcript storage functions
from transcript_storage import TranscriptVectorizer, process_transcript

# Load environment variables from .env file
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description='Store a transcript in Qdrant')
    parser.add_argument('--file', '-f', type=str, required=True, 
                       help='Path to the transcript JSON file')
    parser.add_argument('--id', type=str, 
                       help='Unique ID for this transcript (defaults to filename without extension)')
    parser.add_argument('--title', '-t', type=str, 
                       help='Title for this transcript (defaults to filename without extension)')
    parser.add_argument('--collection', '-c', type=str, 
                       help='Name of the Qdrant collection (defaults to env variable or "meeting_transcripts")')
    parser.add_argument('--model', '-m', type=str, 
                       help='Sentence transformer model to use for embeddings (defaults to env variable or "all-MiniLM-L6-v2")')
    
    args = parser.parse_args()
    
    # Validate the file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        return
    
    # Generate meeting ID and title if not provided
    filename = os.path.basename(args.file)
    base_name = os.path.splitext(filename)[0]
    
    meeting_id = args.id if args.id else base_name
    meeting_title = args.title if args.title else base_name.replace("_", " ").title()
    
    # Get collection and model from environment variables if not provided
    collection_name = args.collection or os.getenv("QDRANT_COLLECTION", "meeting_transcripts")
    embedding_model = args.model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Process the transcript
    print(f"Processing transcript: {args.file}")
    processed_data = process_transcript(args.file)
    
    if not processed_data:
        print("Failed to process transcript")
        return
    
    # Create meeting metadata
    meeting_metadata = {
        "meeting_id": meeting_id,
        "title": meeting_title,
        "file_path": args.file,
        "timestamp": datetime.now().isoformat()
    }
    
    # Initialize the vectorizer (will use env variables for URL and API key)
    vectorizer = TranscriptVectorizer(
        embedding_model=embedding_model,
        collection_name=collection_name
    )
    
    # Store the transcript
    success = vectorizer.process_transcript_data(processed_data, meeting_metadata)
    
    if success:
        print(f"Successfully stored transcript with ID: {meeting_id}")
        print(f"Using Qdrant URL: {os.getenv('QDRANT_URL', 'http://localhost:6333')}")
        print(f"Using collection: {collection_name}")
    else:
        print("Failed to store transcript")

if __name__ == "__main__":
    main()