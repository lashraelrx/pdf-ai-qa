"""PDF processing module for loading and chunking PDF documents."""

import hashlib
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import CHUNK_SIZE, CHUNK_OVERLAP, UPLOADS_DIR


class PDFProcessor:
    """Handles PDF loading, text extraction, and intelligent chunking."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def get_pdf_hash(self, pdf_path: str | Path) -> str:
        """Generate a unique hash for a PDF file based on its content."""
        pdf_path = Path(pdf_path)
        hasher = hashlib.sha256()
        
        with open(pdf_path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()[:16]

    def get_page_count(self, pdf_path: str | Path) -> int:
        """Get the total number of pages in a PDF."""
        pdf_path = Path(pdf_path)
        with fitz.open(pdf_path) as doc:
            return len(doc)

    def extract_text_by_page(self, pdf_path: str | Path) -> Generator[tuple[int, str], None, None]:
        """
        Extract text from each page of a PDF.
        
        Yields:
            Tuple of (page_number, text) for each page (1-indexed)
        """
        pdf_path = Path(pdf_path)
        
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                if text.strip():
                    yield page_num, text

    def process_pdf(
        self, 
        pdf_path: str | Path,
        progress_callback: callable = None
    ) -> list[Document]:
        """
        Process a PDF file into chunks with page metadata.
        
        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback function(current, total) for progress updates
            
        Returns:
            List of Document objects with text and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"Invalid file type. Expected PDF, got: {pdf_path.suffix}")

        pdf_hash = self.get_pdf_hash(pdf_path)
        total_pages = self.get_page_count(pdf_path)
        
        all_chunks = []
        
        for page_num, page_text in self.extract_text_by_page(pdf_path):
            # Create chunks for this page
            page_chunks = self.text_splitter.split_text(page_text)
            
            # Create Document objects with metadata
            for chunk_idx, chunk_text in enumerate(page_chunks):
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": pdf_path.name,
                        "page_number": page_num,
                        "chunk_index": chunk_idx,
                        "pdf_hash": pdf_hash,
                        "total_pages": total_pages
                    }
                )
                all_chunks.append(doc)
            
            # Report progress
            if progress_callback:
                progress_callback(page_num, total_pages)
        
        return all_chunks

    def save_uploaded_pdf(self, file_path: str, original_name: str) -> Path:
        """
        Save an uploaded PDF to the uploads directory.
        
        Args:
            file_path: Temporary path of the uploaded file
            original_name: Original filename
            
        Returns:
            Path to the saved file
        """
        dest_path = UPLOADS_DIR / original_name
        
        # Handle duplicate names
        counter = 1
        while dest_path.exists():
            stem = Path(original_name).stem
            suffix = Path(original_name).suffix
            dest_path = UPLOADS_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # Copy file
        import shutil
        shutil.copy2(file_path, dest_path)
        
        return dest_path

