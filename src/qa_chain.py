"""QA Chain module for LangChain question-answering setup."""

import logging
import requests
from typing import Optional

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .config import OLLAMA_BASE_URL, LLM_MODEL, TOP_K_RESULTS
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


# Custom prompt template for QA
QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the context provided above
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but comprehensive
- Reference specific details from the context when possible

Answer:"""

# Translation templates
TRANSLATE_TO_EN_TEMPLATE = """Translate the following Turkish text to English. Only return the translation, no other text.

Text: {text}

Translation:"""

TRANSLATE_TO_TR_TEMPLATE = """Translate the following English text to Turkish. Only return the translation, no other text.

Text: {text}

Translation:"""


class QAChain:
    """Handles question-answering using LangChain with Ollama."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self._llm = None
        self._chain = None
        self._translator_chain = None

    @staticmethod
    def check_ollama_running() -> tuple[bool, str]:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            Tuple of (is_running, message)
        """
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                return True, "Ollama is running"
            return False, f"Ollama returned status code: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Please ensure Ollama is running."
        except requests.exceptions.Timeout:
            return False, "Connection to Ollama timed out"
        except Exception as e:
            return False, f"Error checking Ollama: {str(e)}"

    @staticmethod
    def check_model_available(model_name: str) -> tuple[bool, str]:
        """
        Check if a specific model is available in Ollama.
        
        Returns:
            Tuple of (is_available, message)
        """
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                
                # Check for exact match or base name match
                model_base = model_name.split(":")[0]
                if model_base in model_names or model_name in [m.get("name") for m in models]:
                    return True, f"Model {model_name} is available"
                
                return False, f"Model {model_name} not found. Available models: {', '.join(model_names) or 'none'}. Run 'ollama pull {model_name}' to download it."
            return False, "Could not fetch model list from Ollama"
        except Exception as e:
            return False, f"Error checking model: {str(e)}"

    @property
    def llm(self) -> OllamaLLM:
        """Lazy initialization of the LLM."""
        if self._llm is None:
            self._llm = OllamaLLM(
                base_url=OLLAMA_BASE_URL,
                model=LLM_MODEL,
                temperature=0.1
            )
        return self._llm

    @property
    def chain(self):
        """Get or create the QA chain using LCEL."""
        if self._chain is None:
            prompt = PromptTemplate(
                template=QA_PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            # Use LCEL (LangChain Expression Language)
            self._chain = prompt | self.llm | StrOutputParser()
        return self._chain

    def translate(self, text: str, target_lang: str = "en") -> str:
        """
        Translate text using the LLM.
        
        Args:
            text: Text to translate
            target_lang: Target language code ('en' or 'tr')
            
        Returns:
            Translated text
        """
        if not text.strip():
            return ""
            
        template = TRANSLATE_TO_EN_TEMPLATE if target_lang == "en" else TRANSLATE_TO_TR_TEMPLATE
        prompt = PromptTemplate(template=template, input_variables=["text"])
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            return chain.invoke({"text": text}).strip()
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    def format_context(self, documents: list[Document]) -> tuple[str, list[int]]:
        """
        Format retrieved documents into a context string.
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Tuple of (formatted_context, list_of_page_numbers)
        """
        if not documents:
            return "", []

        context_parts = []
        page_numbers = set()
        
        for i, doc in enumerate(documents, 1):
            page_num = doc.metadata.get("page_number", "?")
            source = doc.metadata.get("source", "unknown")
            
            context_parts.append(f"[Source: {source}, Page {page_num}]\n{doc.page_content}")
            
            if isinstance(page_num, int):
                page_numbers.add(page_num)
        
        context = "\n\n---\n\n".join(context_parts)
        return context, sorted(page_numbers)

    def ask(
        self,
        question: str,
        pdf_hash: Optional[str] = None,
        k: int = TOP_K_RESULTS,
        translate_to_english: bool = False
    ) -> dict:
        """
        Ask a question and get an answer based on the stored documents.
        
        Args:
            question: The question to ask
            pdf_hash: Optional filter to search only in a specific PDF
            k: Number of context chunks to retrieve
            translate_to_english: Whether to translate query from TR to EN and answer back to TR
            
        Returns:
            Dictionary with 'answer', 'source_pages', 'context_chunks', and any errors
        """
        result = {
            "answer": "",
            "original_question": question,
            "translated_question": None,
            "original_answer": None,
            "source_pages": [],
            "context_chunks": [],
            "error": None
        }

        # Check Ollama is running
        is_running, message = self.check_ollama_running()
        if not is_running:
            result["error"] = message
            return result

        # Check required models
        for model in [LLM_MODEL, "nomic-embed-text"]:
            is_available, message = self.check_model_available(model)
            if not is_available:
                result["error"] = message
                return result

        try:
            search_query = question
            
            # Translate query if needed
            if translate_to_english:
                search_query = self.translate(question, "en")
                result["translated_question"] = search_query

            # Retrieve relevant documents
            documents = self.vector_store.similarity_search(
                query=search_query,
                k=k,
                pdf_hash=pdf_hash
            )
            
            if not documents:
                result["answer"] = "No relevant information found in the uploaded documents. Please make sure a PDF has been processed."
                return result

            # Format context
            context, page_numbers = self.format_context(documents)
            result["source_pages"] = page_numbers
            result["context_chunks"] = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "page": doc.metadata.get("page_number", "?"),
                    "source": doc.metadata.get("source", "unknown")
                }
                for doc in documents
            ]

            # Generate answer using LCEL chain
            response = self.chain.invoke({
                "context": context,
                "question": search_query
            })
            
            answer_text = response.strip()
            result["original_answer"] = answer_text
            
            # Translate answer back if needed
            if translate_to_english:
                result["answer"] = self.translate(answer_text, "tr")
            else:
                result["answer"] = answer_text
            
        except Exception as e:
            logger.error(f"Error during QA: {e}")
            result["error"] = f"Error generating answer: {str(e)}"

        return result

    def get_system_status(self) -> dict:
        """
        Get the current system status including Ollama and model availability.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "ollama_running": False,
            "llm_model_available": False,
            "embedding_model_available": False,
            "messages": []
        }

        # Check Ollama
        is_running, message = self.check_ollama_running()
        status["ollama_running"] = is_running
        status["messages"].append(message)

        if is_running:
            # Check LLM model
            is_available, message = self.check_model_available(LLM_MODEL)
            status["llm_model_available"] = is_available
            status["messages"].append(message)

            # Check embedding model
            is_available, message = self.check_model_available("nomic-embed-text")
            status["embedding_model_available"] = is_available
            status["messages"].append(message)

        # Get vector store stats
        status["vector_store"] = self.vector_store.get_collection_stats()

        return status
