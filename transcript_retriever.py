import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range

# Load environment variables
load_dotenv()

class TranscriptRetriever:
    """
    A class to retrieve relevant transcript segments from Qdrant
    based on semantic similarity to a query.
    """
    
    def __init__(
        self,
        embedding_model: str = None,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = None
    ):
        """
        Initialize the retriever with embedding model and database connection.
        
        Args:
            embedding_model: Name of the sentence-transformers model to use
            qdrant_url: URL of the Qdrant server
            qdrant_api_key: API key for Qdrant
            collection_name: Name of the collection to search
        """
        # Get configuration from environment or use provided values
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "meeting_transcripts")
        
        # Set up the embedding model
        self.model = SentenceTransformer(self.embedding_model)
        
        # Set up Qdrant client
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        
        # Ensure collection exists
        self._check_collection()
    
    def _check_collection(self):
        """Verify that the collection exists"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            raise ValueError(f"Collection '{self.collection_name}' not found in Qdrant. Available collections: {collection_names}")
    
    def retrieve(
        self, 
        query: str, 
        limit: int = 3,
        threshold: float = 0.4,
        meeting_id: Optional[str] = None,
        speakers: Optional[List[str]] = None,
        time_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant transcript segments based on a query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            threshold: Minimum score threshold (0-1, higher is more similar)
            meeting_id: Optional meeting ID to filter by
            speakers: Optional list of speakers to filter by
            time_range: Optional tuple of (start_time, end_time) to filter by
            
        Returns:
            List of relevant transcript segments with metadata
        """
        # Generate embedding for the query
        query_vector = self.model.encode(query).tolist()
        
        # Build filter conditions
        filter_conditions = []
        
        if meeting_id:
            filter_conditions.append(
                FieldCondition(
                    key="meeting_id",
                    match=MatchValue(value=meeting_id)
                )
            )
        
        if speakers:
            # Find segments where any of the specified speakers are present
            filter_conditions.append(
                FieldCondition(
                    key="speakers",
                    match=MatchValue(value=speakers)
                )
            )
        
        # Create filter if any conditions exist
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        # Perform the search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            # score_threshold=0.4,
            query_filter=search_filter
        )
        
        # Format the results
        formatted_results = []
        for result in search_results:
            # Extract relevant data
            formatted_result = {
                "score": result.score,
                "meeting_id": result.payload.get("meeting_id", ""),
                "title": result.payload.get("title", ""),
                "speakers": result.payload.get("speakers", []),
                "segments": result.payload.get("segments", []),
                "text": result.payload.get("text", ""),
                "timestamp": result.payload.get("timestamp", "")
            }
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def format_for_llm(self, results: List[Dict[str, Any]], detailed: bool = False) -> str:
        """
        Format retrieval results for passing to an LLM.
        
        Args:
            results: List of retrieval results
            detailed: Whether to include detailed segment information
            
        Returns:
            Formatted context string for the LLM
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, result in enumerate(results):
            context_parts.append(f"--- Transcript {i+1} (Relevance: {result['score']:.2f}) ---")
            context_parts.append(f"Meeting: {result['title']}")
            
            if detailed:
                # Include individual speaker segments
                for j, segment in enumerate(result.get('segments', [])):
                    context_parts.append(f"{segment.get('speaker')}: {segment.get('text')}")
            else:
                # Just include the full text
                context_parts.append(result.get('text', ''))
            
            context_parts.append("")  # Empty line between results
        
        return "\n".join(context_parts)

    def generate_llm_prompt(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate a complete prompt for the LLM with the query and retrieved context.
        
        Args:
            query: The user's question
            results: The retrieved context
            
        Returns:
            Formatted prompt for the LLM
        """
        context = self.format_for_llm(results, detailed=True)
        
        prompt = f"""
You are a helpful AI assistant that answers questions about meetings based on transcript excerpts.
Use only the provided transcript excerpts to answer the question below.
If the answer cannot be determined from the provided information, say so clearly.

TRANSCRIPT EXCERPTS:
{context}

USER QUESTION: {query}

Please provide a clear, concise answer based only on the information in the transcript excerpts.
"""
        return prompt