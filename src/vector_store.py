import logging
import time
import uuid
import re
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from .config import (
    CHROMA_DIR,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_REQUEST_DELAY,
)
from ollama import ResponseError

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Sanitize text to prevent embedding failures."""
    # Remove null bytes
    text = text.replace('\x00', '')
    # Remove other control characters except newlines/tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Remove surrogate pairs that cause issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    # Normalize excessive whitespace (but keep single newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class VectorStore:
    """Handles ChromaDB operations for document storage and retrieval."""

    def __init__(self, collection_name: str = "pdf_documents"):
        self.collection_name = collection_name
        self.chroma_dir = str(CHROMA_DIR)
        self._embeddings = None
        self._vector_store = None
        self._client = None

    @property
    def embeddings(self) -> OllamaEmbeddings:
        """Lazy initialization of embeddings model."""
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=EMBEDDING_MODEL
            )
        return self._embeddings

    @property
    def client(self) -> chromadb.PersistentClient:
        """Get or create ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
        return self._client

    def get_vector_store(self) -> Chroma:
        """Get or create the vector store."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.chroma_dir
            )
        return self._vector_store

    def check_pdf_exists(self, pdf_hash: str) -> bool:
        """
        Check if a PDF has already been processed and stored.
        
        Args:
            pdf_hash: The unique hash of the PDF
            
        Returns:
            True if the PDF exists in the store
        """
        try:
            collection = self.client.get_or_create_collection(self.collection_name)
            results = collection.get(
                where={"pdf_hash": pdf_hash},
                limit=1
            )
            return len(results["ids"]) > 0
        except Exception as e:
            logger.warning(f"Error checking PDF existence: {e}")
            return False

    def add_documents(
        self,
        documents: list[Document],
        progress_callback: callable = None
    ) -> int:
        """
        Add documents to the vector store in batches.
        
        Args:
            documents: List of Document objects to add
            progress_callback: Optional callback function(current, total, status) for progress updates
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        vector_store = self.get_vector_store()
        collection = vector_store._collection  # direct access for fine-grained inserts
        total_docs = len(documents)
        processed = 0
        
        # Process in batches
        for i in range(0, total_docs, EMBEDDING_BATCH_SIZE):
            batch = documents[i:i + EMBEDDING_BATCH_SIZE]

            # === DIAGNOSTIC LOGGING ===
            for j, doc in enumerate(batch):
                chunk_idx = i + j
                content = doc.page_content
                content_len = len(content)
                
                # Check for problematic content
                has_null = '\x00' in content
                has_weird = any(ord(c) > 65535 for c in content)
                non_ascii_count = sum(1 for c in content if ord(c) > 127)
                
                logger.info(
                    f"Chunk {chunk_idx}: {content_len} chars, "
                    f"page {doc.metadata.get('page', '?')}, "
                    f"null={has_null}, weird_chars={has_weird}, "
                    f"non_ascii={non_ascii_count}"
                )
                
                # Log preview of chunks in the failing batch range (24-31 based on your error)
                if 20 <= chunk_idx <= 31:
                    logger.info(f"Chunk {chunk_idx} PREVIEW: {repr(content[:300])}")
            # === END DIAGNOSTIC ===

            # Clean texts before embedding
            texts = [clean_text(doc.page_content) for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            ids = [
                f"{doc.metadata.get('pdf_hash', 'pdf')}-{uuid.uuid4().hex}"
                for doc in batch
            ]

            # Skip empty texts after cleaning
            valid_indices = [idx for idx, text in enumerate(texts) if text.strip()]
            if len(valid_indices) < len(texts):
                logger.warning(f"Skipping {len(texts) - len(valid_indices)} empty chunks after cleaning")
                texts = [texts[idx] for idx in valid_indices]
                metadatas = [metadatas[idx] for idx in valid_indices]
                ids = [ids[idx] for idx in valid_indices]
            
            if not texts:
                logger.warning(f"Batch {i//EMBEDDING_BATCH_SIZE} has no valid texts, skipping")
                continue

            for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
                try:
                    logger.info(f"Attempting embedding for batch {i//EMBEDDING_BATCH_SIZE + 1}, chunks {i}-{i+len(batch)-1}")
                    embeddings = self.embeddings.embed_documents(texts)
                    collection.add(
                        ids=ids,
                        documents=texts,
                        metadatas=metadatas,
                        embeddings=embeddings,
                    )
                    logger.info(f"Successfully embedded batch {i//EMBEDDING_BATCH_SIZE + 1}")
                    break
                except ResponseError as err:
                    logger.warning(
                        "Embedding request failed (attempt %s/%s): %s",
                        attempt,
                        EMBEDDING_MAX_RETRIES,
                        err,
                    )
                    if attempt == EMBEDDING_MAX_RETRIES:
                        # Log the problematic texts for debugging
                        logger.error(f"Failed batch texts lengths: {[len(t) for t in texts]}")
                        for idx, t in enumerate(texts):
                            logger.error(f"Text {idx} preview: {repr(t[:200])}")
                        raise RuntimeError(
                            "Embedding failed repeatedly. "
                            f"Please restart Ollama and ensure the '{EMBEDDING_MODEL}' model is installed."
                        ) from err
                    time.sleep(2 * attempt)
                except Exception as err:  # catch-all to bubble up meaningful message
                    logger.error("Unexpected error while adding documents: %s", err)
                    raise
            
            processed += len(batch)
            if progress_callback:
                progress_callback(processed, total_docs, f"Embedding chunks {processed}/{total_docs}")

            if EMBEDDING_REQUEST_DELAY > 0:
                time.sleep(EMBEDDING_REQUEST_DELAY)
        
        return total_docs

    def similarity_search(
        self,
        query: str,
        k: int = TOP_K_RESULTS,
        pdf_hash: Optional[str] = None
    ) -> list[Document]:
        vector_store = self.get_vector_store()
        
        filter_dict = None
        if pdf_hash:
            filter_dict = {"pdf_hash": pdf_hash}
        
        # Use MMR for diverse results across the document
        results = vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=k * 4,  # Fetch more candidates for diversity
            lambda_mult=0.7,  # Balance relevance vs diversity (0.5-0.8 works well)
            filter=filter_dict
        )
        
        return results

    def delete_pdf_documents(self, pdf_hash: str) -> int:
        """
        Delete all documents associated with a specific PDF.
        
        Args:
            pdf_hash: The unique hash of the PDF to delete
            
        Returns:
            Number of documents deleted
        """
        try:
            collection = self.client.get_or_create_collection(self.collection_name)
            
            # Get all IDs with this hash
            results = collection.get(
                where={"pdf_hash": pdf_hash}
            )
            
            if results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])
            
            return 0
        except Exception as e:
            logger.error(f"Error deleting PDF documents: {e}")
            return 0

    def clear_all(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.client.delete_collection(self.collection_name)
            self._vector_store = None
            logger.info("Vector store cleared")
        except Exception as e:
            logger.warning(f"Error clearing vector store: {e}")

    def get_collection_stats(self) -> dict:
        """Get statistics about the current collection."""
        try:
            collection = self.client.get_or_create_collection(self.collection_name)
            count = collection.count()
            
            # Get unique PDFs
            if count > 0:
                results = collection.get(include=["metadatas"])
                unique_pdfs = set()
                for meta in results["metadatas"]:
                    if meta and "pdf_hash" in meta:
                        unique_pdfs.add(meta.get("source", "unknown"))
                
                return {
                    "total_chunks": count,
                    "unique_pdfs": len(unique_pdfs),
                    "pdf_names": list(unique_pdfs)
                }
            
            return {"total_chunks": 0, "unique_pdfs": 0, "pdf_names": []}
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"total_chunks": 0, "unique_pdfs": 0, "pdf_names": [], "error": str(e)}








