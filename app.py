"""
PDF Question-Answering System with RAG

A local application for asking questions about PDF documents using
Retrieval-Augmented Generation with Ollama and ChromaDB.
"""

import logging
from pathlib import Path
from typing import Optional
import tempfile
import shutil

import streamlit as st

from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.qa_chain import QAChain
from src.config import LLM_MODEL, EMBEDDING_MODEL, UPLOADS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="PDF Q&A System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "pdf_processor" not in st.session_state:
    st.session_state.pdf_processor = PDFProcessor()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = QAChain(st.session_state.vector_store)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_pdf_hash" not in st.session_state:
    st.session_state.current_pdf_hash = None
if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "translate_enabled" not in st.session_state:
    st.session_state.translate_enabled = False


def check_system_status():
    """Check and return the current system status."""
    try:
        status = st.session_state.qa_chain.get_system_status()
        return status
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        return {"ollama_running": False, "error": str(e)}


def process_uploaded_pdf(uploaded_file):
    """Process an uploaded PDF file."""
    if uploaded_file is None:
        return False, "Please upload a PDF file first."
    
    # Check system status
    status = check_system_status()
    if not status.get("ollama_running"):
        return False, "❌ Ollama is not running. Please start Ollama and try again."
    
    if not status.get("embedding_model_available"):
        return False, f"❌ Embedding model not found. Run: ollama pull {EMBEDDING_MODEL}"
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = Path(tmp_file.name)
        
        original_name = uploaded_file.name
        
        # Get PDF hash
        pdf_hash = st.session_state.pdf_processor.get_pdf_hash(tmp_path)
        
        # Check if already processed
        if st.session_state.vector_store.check_pdf_exists(pdf_hash):
            st.session_state.current_pdf_hash = pdf_hash
            st.session_state.current_pdf_name = original_name
            st.session_state.pdf_processed = True
            page_count = st.session_state.pdf_processor.get_page_count(tmp_path)
            tmp_path.unlink()  # Clean up temp file
            return True, f"✅ PDF '{original_name}' ({page_count} pages) already processed. Ready to answer questions!"
        
        # Save PDF permanently
        dest_path = UPLOADS_DIR / original_name
        shutil.copy2(tmp_path, dest_path)
        tmp_path.unlink()  # Clean up temp file
        
        # Process PDF
        progress_bar = st.progress(0, text="Processing PDF...")
        
        def progress_callback(current, total):
            progress_bar.progress(current / total, text=f"Processing page {current}/{total}")
        
        chunks = st.session_state.pdf_processor.process_pdf(dest_path, progress_callback=progress_callback)
        
        if not chunks:
            return False, "⚠️ No text could be extracted from the PDF."
        
        # Add to vector store
        progress_bar.progress(0.5, text="Creating embeddings...")
        
        def embedding_callback(current, total, msg):
            progress_bar.progress(0.5 + (current / total) * 0.5, text=msg)
        
        num_added = st.session_state.vector_store.add_documents(chunks, progress_callback=embedding_callback)
        
        progress_bar.progress(1.0, text="Complete!")
        progress_bar.empty()
        
        st.session_state.current_pdf_hash = pdf_hash
        st.session_state.current_pdf_name = original_name
        st.session_state.pdf_processed = True
        
        page_count = chunks[0].metadata.get("total_pages", "?") if chunks else "?"
        return True, f"✅ Successfully processed '{original_name}'!\n📄 {page_count} pages → {num_added} chunks embedded"
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        return False, f"❌ Error processing PDF: {str(e)}"


def ask_question(question: str):
    """Ask a question about the processed PDF."""
    if not question.strip():
        return None
    
    try:
        result = st.session_state.qa_chain.ask(
            question, 
            pdf_hash=st.session_state.current_pdf_hash,
            translate_to_english=st.session_state.translate_enabled
        )
        return result
    except Exception as e:
        logger.error(f"Error asking question: {e}", exc_info=True)
        return {"error": str(e), "answer": "", "source_pages": [], "context_chunks": []}


def reset_system():
    """Reset the entire system."""
    try:
        st.session_state.vector_store.clear_all()
        st.session_state.chat_history = []
        st.session_state.current_pdf_hash = None
        st.session_state.current_pdf_name = None
        st.session_state.pdf_processed = False
        return True, "🔄 System reset complete."
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


# Main UI
st.title("📚 PDF Question-Answering System")
st.markdown("Upload a PDF and ask questions using local AI with RAG")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    
    status = check_system_status()
    
    if status.get("ollama_running"):
        st.success("✅ Ollama is running")
        
        if status.get("llm_model_available"):
            st.success(f"✅ LLM: {LLM_MODEL}")
        else:
            st.error(f"❌ LLM not found")
            st.code(f"ollama pull {LLM_MODEL}")
        
        if status.get("embedding_model_available"):
            st.success(f"✅ Embeddings: {EMBEDDING_MODEL}")
        else:
            st.error(f"❌ Embeddings not found")
            st.code(f"ollama pull {EMBEDDING_MODEL}")
    else:
        st.error("❌ Ollama is not running")
        st.info("Start Ollama first: `ollama serve`")
    
    # Vector store stats
    vs_stats = status.get("vector_store", {})
    st.divider()
    st.subheader("📊 Vector Store")
    if vs_stats.get("total_chunks", 0) > 0:
        st.info(f"📄 {vs_stats['total_chunks']} chunks\n🗂️ {vs_stats['unique_pdfs']} PDF(s)")
    else:
        st.info("Empty - upload a PDF to get started")
    
    st.divider()
    
    # Translation toggle
    st.subheader("🌐 Translation")
    st.session_state.translate_enabled = st.toggle(
        "Enable TR ↔ EN Translation", 
        value=st.session_state.translate_enabled,
        help="Translate Turkish queries to English and answers back to Turkish"
    )
    
    st.divider()
    if st.button("🗑️ Reset System", type="secondary", use_container_width=True):
        success, msg = reset_system()
        if success:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📤 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to analyze"
    )
    
    if uploaded_file is not None:
        if st.button("🔄 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                success, message = process_uploaded_pdf(uploaded_file)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    if st.session_state.current_pdf_name:
        st.info(f"📖 Current: {st.session_state.current_pdf_name}")

with col2:
    st.header("💬 Ask Questions")
    
    # Chat history display
    chat_container = st.container(height=400)
    
    with chat_container:
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                # Display original question
                st.write(q)
                # If translation was used, show the translated query in a caption
                if a.get("translated_question"):
                    st.caption(f"Translated to EN: {a['translated_question']}")
            
            with st.chat_message("assistant"):
                st.write(a["answer"])
                
                # If translation was used, show original English answer
                if a.get("original_answer") and a["original_answer"] != a["answer"]:
                    with st.expander("Original English Answer"):
                        st.write(a["original_answer"])
                
                if a.get("source_pages"):
                    st.caption(f"📖 Sources: Pages {', '.join(map(str, a['source_pages']))}")
    
    # Question input
    if st.session_state.pdf_processed:
        question = st.chat_input("Ask a question about the PDF...")
        
        if question:
            # Add user message
            with chat_container:
                with st.chat_message("user"):
                    st.write(question)
            
            # Get answer
            with st.spinner("Thinking..."):
                result = ask_question(question)
            
            if result:
                if result.get("error"):
                    st.error(f"❌ {result['error']}")
                else:
                    # Add to history
                    st.session_state.chat_history.append((question, result))
                    
                    # Display answer
                    with chat_container:
                        # If translated, show translated query note
                        if result.get("translated_question"):
                            st.caption(f"Translated to EN: {result['translated_question']}")
                            
                        with st.chat_message("assistant"):
                            st.write(result["answer"])
                            
                            if result.get("original_answer") and result["original_answer"] != result["answer"]:
                                with st.expander("Original English Answer"):
                                    st.write(result["original_answer"])
                                    
                            if result.get("source_pages"):
                                st.caption(f"📖 Sources: Pages {', '.join(map(str, result['source_pages']))}")
                    
                    # Removed st.rerun() here to prevent clearing the input too fast or double-submission issues
                    # Streamlit handles state updates automatically
    else:
        st.info("👆 Upload and process a PDF first to start asking questions")
    
    # Show context chunks in expander
    if st.session_state.chat_history:
        with st.expander("📄 View Retrieved Context", expanded=False):
            last_result = st.session_state.chat_history[-1][1]
            if last_result.get("context_chunks"):
                for i, chunk in enumerate(last_result["context_chunks"], 1):
                    st.markdown(f"**Chunk {i}** (Page {chunk['page']})")
                    st.text(chunk["content"])
                    st.divider()

# Footer
st.divider()
st.caption("Powered by Ollama, LangChain, and ChromaDB")
