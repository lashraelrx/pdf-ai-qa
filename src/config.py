"""Configuration constants for the PDF QA system."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "mxbai-embed-large"

# Chunking settings tuned for large technical PDFs
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Retrieval settings
TOP_K_RESULTS = 15

# Batch processing / throttling
# Smaller batches avoid overloading the Ollama embedding endpoint on huge PDFs.
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 1))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", 3))
EMBEDDING_REQUEST_DELAY = float(os.getenv("EMBEDDING_REQUEST_DELAY", 1))

