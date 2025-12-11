# PDF Question-Answering System with RAG
A local PDF question-answering application using Retrieval-Augmented Generation (RAG).

## Tech Stack

- **Python 3.10+**
- **Ollama** - Local LLM inference (llama3.1:8b) and embeddings (nomic-embed-text)
- **LangChain** - LLM orchestration framework
- **ChromaDB** - Vector database for semantic search
- **PyMuPDF (fitz)** - PDF text extraction
- **Streamlit** - Web UI framework

## Project Structure

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

## To Start

### Prerequisites

1. **Install Ollama**

2. **Pull required models**:

3. **Run the application**:
   ```bash
   app.py
   ```
   
MIT License - feel free to use and modify as needed.

