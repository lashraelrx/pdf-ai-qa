# 📚 PDF Question-Answering System with RAG

A local PDF question-answering application using Retrieval-Augmented Generation (RAG) architecture. Ask questions about your PDF documents and get accurate answers with source citations.

## ✨ Features

- **Local & Private**: All processing happens locally using Ollama - no data leaves your machine
- **Large PDF Support**: Efficiently handles PDFs up to 3000+ pages
- **Smart Chunking**: Uses RecursiveCharacterTextSplitter for intelligent text segmentation
- **Persistent Storage**: ChromaDB persists embeddings - re-uploading the same PDF skips processing
- **Source Citations**: Shows which pages the answer was derived from
- **Progress Tracking**: Real-time progress updates for large PDF processing
- **Modern UI**: Clean Gradio interface with chat-style interactions

## 🛠️ Tech Stack

- **Python 3.10+**
- **Ollama** - Local LLM inference (llama3.1:8b) and embeddings (nomic-embed-text)
- **LangChain** - LLM orchestration framework
- **ChromaDB** - Vector database for semantic search
- **PyMuPDF (fitz)** - Fast PDF text extraction
- **Streamlit** - Web UI framework

## 📁 Project Structure

```
pdf-qa/
├── app.py              # Main Gradio application
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py    # PDF loading and chunking
│   ├── vector_store.py     # ChromaDB operations
│   ├── qa_chain.py         # LangChain QA setup
│   └── config.py           # Configuration constants
├── data/
│   └── uploads/            # Uploaded PDFs
├── chroma_db/              # Persisted vector database
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)

2. **Pull required models**:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

3. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd pdf-qa
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to `http://localhost:7860`

## 📖 Usage

1. **Check System Status**: The sidebar shows whether Ollama and required models are available

2. **Upload a PDF**: Click the upload area or drag and drop a PDF file

3. **Process the PDF**: Click "Process PDF" and wait for completion
   - Progress bar shows extraction and embedding progress
   - If the PDF was previously processed, embeddings are loaded from cache

4. **Ask Questions**: Type your question in the text box and press Enter or click "Ask"
   - The AI will search relevant chunks and generate an answer
   - Source pages are displayed below the chat
   - Expand "Retrieved Context Chunks" to see the exact text used

5. **Clear/Reset**: 
   - "Clear Chat" removes conversation history
   - "Reset System" clears the vector database to start fresh

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.1:8b"
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking settings
CHUNK_SIZE = 1000        # Characters per chunk
CHUNK_OVERLAP = 200      # Overlap between chunks

# Retrieval settings
TOP_K_RESULTS = 5        # Number of context chunks to retrieve

# Batch processing
EMBEDDING_BATCH_SIZE = 100  # Chunks per embedding batch
```

## 🔧 Troubleshooting

### "Ollama is not running"
- Start Ollama: `ollama serve`
- Check if running: `curl http://localhost:11434/api/tags`

### "Model not found"
- Pull the required model: `ollama pull llama3.1:8b`
- List available models: `ollama list`

### Slow or failing processing on huge PDFs
- Large PDFs now stream through smaller embedding batches (default 4 chunks/request)
- You can tune behaviour via environment variables:
  ```bash
  set CHUNK_SIZE=1500
  set EMBEDDING_BATCH_SIZE=2
  set EMBEDDING_REQUEST_DELAY=0.2
  ```
- Restart the app after tweaking values

### Out of memory
- Reduce `CHUNK_SIZE` in config
- Use a smaller LLM model
- Close other applications

## 🔒 Privacy

All processing is done locally:
- PDFs are stored in `data/uploads/`
- Embeddings are stored in `chroma_db/`
- No data is sent to external services
- Delete these folders to remove all traces

## 📝 License

MIT License - feel free to use and modify as needed.

