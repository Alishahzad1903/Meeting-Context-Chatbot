import json
import os
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime

# For environment variables
from dotenv import load_dotenv

# For embeddings
from sentence_transformers import SentenceTransformer

# For Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


# Load environment variables from .env file
load_dotenv()


class TranscriptVectorizer:
    """
    A class that handles creating embeddings from transcript data
    and storing in a Qdrant vector database. Each transcript is
    treated as a single chunk.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "meeting_transcripts"
    ):
        """
        Initialize the vectorizer with embedding model and database connection.
        
        Args:
            embedding_model: Name of the sentence-transformers model to use
            qdrant_url: URL of the Qdrant server (overrides env variable)
            qdrant_api_key: API key for Qdrant (overrides env variable)
            collection_name: Name of the collection to store vectors
        """
        # Set up the embedding model
        self.model = SentenceTransformer(embedding_model)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        
        # Get Qdrant URL and API key from environment variables if not provided
        qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        # Set up Qdrant client
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        
        # Ensure the collection exists
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Create the collection if it doesn't already exist"""
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            # Create the collection with the correct vector size
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE  # Using cosine similarity
                )
            )
            print(f"Created new collection: {self.collection_name}")
    
    def _extract_text_and_speakers(self, segments: List[Dict]) -> Dict:
        """
        Extract just text and speaker information from segments.
        Each file's transcript is processed as a single chunk.
        
        Args:
            segments: List of transcript segments/utterances
            
        Returns:
            Dictionary with combined text and processed segments
        """
        # Extract just text and speaker info
        processed_segments = [
            {
                "text": seg["text"],
                "speaker": seg["speaker"]
            } for seg in segments
        ]
        
        # Combine texts for the embedding
        combined_text = " ".join([seg["text"] for seg in processed_segments])
        
        # Create a result with the combined text and segment info
        result = {
            "text": combined_text,
            "segments": processed_segments
        }
        
        return result
    
    def process_transcript_data(self, data: Dict, meeting_metadata: Dict) -> bool:
        """
        Process transcript data and store in Qdrant.
        
        Args:
            data: Dictionary containing transcript data
            meeting_metadata: Metadata about the meeting
            
        Returns:
            Boolean indicating success
        """
        # Extract segments from the transcript data
        segments = data.get("segments", [])
        
        if not segments:
            print("Warning: No segments found in transcript data")
            return False
        
        # Process the transcript (entire data as one chunk)
        processed_data = self._extract_text_and_speakers(segments)
        
        # Create embedding for the chunk
        embedding = self.model.encode(processed_data["text"])
        
        # Extract all speakers in this chunk
        speakers = list(set(segment["speaker"] for segment in processed_data["segments"]))
        
        # Prepare the payload with metadata
        payload = {
            "meeting_id": meeting_metadata["meeting_id"],
            "title": meeting_metadata.get("title", "Untitled Meeting"),
            "text": processed_data["text"],
            "speakers": speakers,
            "segments": processed_data["segments"],
            "timestamp": meeting_metadata.get("timestamp", "")
        }
        
        # Add any additional metadata
        for key, value in meeting_metadata.items():
            if key not in payload:
                payload[key] = value
        
        # Create a unique ID for this chunk (using meeting_id directly)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, meeting_metadata["meeting_id"]))
        
        # Create the point
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload=payload
        )
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        print(f"Uploaded transcript data with ID {meeting_metadata['meeting_id']} as a single chunk")
        return True


def process_transcript(transcript_json_path, meeting_id=None, meeting_title=None):
    """
    Process a transcript file and extract just the text and speaker information.
    
    Args:
        transcript_json_path: Path to the transcript JSON file
        meeting_id: Optional ID for the meeting
        meeting_title: Optional title for the meeting
        
    Returns:
        Processed transcript data dictionary
    """
    try:
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {transcript_json_path}: {e}")
        return None
    
    # Extract only the text and speaker from each segment
    processed_data = {
        "segments": []
    }
    
    # Check if the data contains segments
    if "segments" in data:
        # Data already has segments
        segments = data["segments"]
        for segment in segments:
            if "text" in segment and "speaker" in segment:
                processed_data["segments"].append({
                    "text": segment["text"],
                    "speaker": segment["speaker"]
                })
    # For the specific format you provided
    elif "word_segments" in data:
        # Process the transcript based on the format you provided
        for segment in data.get("segments", []):
            if "text" in segment and "speaker" in segment:
                processed_data["segments"].append({
                    "text": segment["text"],
                    "speaker": segment["speaker"]
                })
    else:
        print(f"Warning: Unsupported transcript format in {transcript_json_path}")
        return None
    
    return processed_data