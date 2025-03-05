# Meeting Context Chatbot

A context-aware question answering system for meeting transcripts. This application transcribes audio recordings of meetings, stores the transcripts in a vector database, and enables natural language queries about the meeting content using large language models.

## Features

- **Automatic Transcription**: Convert meeting audio to text with speaker identification
- **Vector Storage**: Store transcripts in Qdrant for semantic search
- **Context Retrieval**: Find relevant meeting segments based on natural language queries
- **LLM Integration**: Use Llama via Groq API to answer questions based on retrieved context
- **User Interface**: Streamlit web app for easy interaction with the system

## System Architecture

![Architecture](https://example.com/architecture.png)

1. **Transcription Pipeline**: Audio → Text + Speaker Identification
2. **Vector Storage**: Text → Embeddings → Qdrant Database
3. **Retrieval System**: Query → Relevant Meeting Segments
4. **LLM Integration**: Context + Query → Natural Language Answer
5. **Web Interface**: User-friendly access to the system

## Installation

### Prerequisites

- Python 3.8+
- Docker (for running Qdrant)
- Hugging Face account (for transcription models)
- Groq API key (for LLM access)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/meeting-context-chatbot.git
cd meeting-context-chatbot
```

2. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```
# Qdrant configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_if_needed
QDRANT_COLLECTION=meeting_transcripts

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Transcription settings
HF_TOKEN=your_huggingface_token

# Groq API
GROQ_API_KEY=your_groq_api_key
```

5. Start Qdrant:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

## Usage

### Transcribe a Meeting

```bash
python transcribe.py --audio_file "path/to/meeting.wav" --hf_token "your_huggingface_token"
```

### Process and Store a Transcript

```bash
python store_chunk.py --file "path/to/transcript.json" --id "meeting-id"
```

### Answer Questions via Command Line

```bash
python answer_with_groq.py "What were the main points discussed in the meeting?"
```

### Launch the Web Interface

```bash
streamlit run streamlit_app.py
```

## API Endpoints

The system includes a FastAPI server with the following endpoints:

### `POST /process-audio`

Process an audio file and store its transcript in the vector database.

**Parameters:**
- `file`: Audio file to transcribe (multipart/form-data)
- `hf_token`: Hugging Face API token (header)
- `min_speakers`: Minimum number of speakers (optional)
- `max_speakers`: Maximum number of speakers (optional)

**Response:**
```json
{
  "status": "success",
  "message": "Audio processed and stored in vector database",
  "transcription_file": "output/meeting_transcript.json"
}
```

### Starting the API Server

```bash
uvicorn app:app --reload
```

## Components

### `transcribe.py`

Transcribes audio recordings using WhisperX with speaker diarization.

### `transcript_storage.py`

Handles creating embeddings and storing them in Qdrant.

### `transcript_retriever.py`

Retrieves relevant meeting segments based on queries.

### `answer_with_groq.py`

Integrates with Groq API to generate answers using Llama.

### `streamlit_app.py`

Provides a web interface for the system.

### `app.py`

FastAPI server for audio processing.

## Troubleshooting

### Common Issues

- **WhisperX not installed**: Run with `--install_deps` flag or install manually
- **Qdrant connection issues**: Check if Qdrant is running and accessible
- **API key errors**: Verify your Groq and Hugging Face API keys
- **Missing dependencies**: Ensure all required packages are installed

## License

[MIT License](LICENSE)

## Acknowledgements

- [WhisperX](https://github.com/m-bain/whisperX) for transcription
- [Qdrant](https://qdrant.tech/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Groq](https://groq.com/) for LLM API access
- [Streamlit](https://streamlit.io/) for the web interface
- [FastAPI](https://fastapi.tiangolo.com/) for the API server
