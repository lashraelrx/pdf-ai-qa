# S1000D PDF Question-Answering System Architecture

This document provides a detailed technical overview of the local RAG (Retrieval-Augmented Generation) system built for querying large technical PDF documents.

## 1. Project Overview

The system allows users to upload PDF documents (specifically designed for large technical manuals like S1000D specs), process them locally, and ask questions in natural language. It features automatic translation (Turkish ↔ English), source citation, and a persistent vector database, all running offline using Ollama.

## 2. System Architecture: RAG Pipeline

The application follows a standard RAG architecture with specific optimizations for local inference:

1.  **Ingestion**: PDF → Text Extraction → Chunking → Embedding → Vector Store.
2.  **Retrieval**: User Query → Embedding → Vector Search (Semantic) → Context Retrieval.
3.  **Generation**: Context + Query → LLM Prompt → Answer Generation.
4.  **Translation Layer**: Optional pre-processing of queries and post-processing of answers.

```mermaid
graph LR
    A[PDF Upload] --> B[PDFProcessor]
    B --> C[Text Chunks]
    C --> D[Ollama Embeddings]
    D --> E[(ChromaDB Vector Store)]
    
    F[User Query (TR/EN)] --> G[QAChain]
    G --> H{Translate?}
    H -- Yes --> I[LLM Translation to EN]
    H -- No --> J[Raw Query]
    
    I & J --> K[Vector Search]
    K --> L[Retrieve Top-K Context]
    E -.-> L
    
    L --> M[LLM Generation (Llama 3.1)]
    M --> N{Translate?}
    N -- Yes --> O[LLM Translation to TR]
    N -- No --> P[Final Answer]
    
    O & P --> Q[UI Display]
```

## 3. Codebase Breakdown

### `app.py` (The Interface)
*   **Role**: Main entry point and User Interface (Streamlit).
*   **Key Functions**:
    *   Manages session state (chat history, current PDF hash).
    *   Handles file uploads and initiates processing.
    *   Displays the chat interface and translation toggles.
    *   Visualizes progress bars for long-running embedding tasks.

### `src/pdf_processor.py` (The ETL Pipeline)
*   **Role**: Extracts text and splits it into manageable pieces.
*   **Key Library**: `PyMuPDF` (fitz) for fast, accurate text extraction.
*   **Algorithm**: `RecursiveCharacterTextSplitter`.
    *   *Why?* It tries to keep paragraphs and sentences together by splitting on `\n\n`, then `\n`, then spaces. This preserves semantic meaning better than fixed-size chopping.
    *   *Settings*: `CHUNK_SIZE=1200`, `CHUNK_OVERLAP=200` (tuned for technical docs to keep context).
*   **Optimization**: Computes a SHA-256 hash of the PDF content to prevent re-processing the same file twice.

### `src/vector_store.py` (The Memory)
*   **Role**: Manages the ChromaDB database.
*   **Key Library**: `langchain-chroma`, `chromadb`.
*   **AI Algorithm**: `OllamaEmbeddings` (nomic-embed-text).
    *   Converts text chunks into dense vectors (arrays of floats).
    *   *Optimization*: Implements **batching and throttling**. Large PDFs crash local inference servers if sent all at once. This file sends chunks in batches of 4 (configurable), waits 0.15s, and retries on failure.
*   **Search**: Performs the actual retrieval of relevant documents using Cosine Similarity.

### `src/qa_chain.py` (The Brain)
*   **Role**: Orchestrates the LLM interactions.
*   **Key Library**: `langchain`, `langchain-ollama`.
*   **AI Approach**:
    *   **Prompt Engineering**: Uses a strict template (`QA_PROMPT_TEMPLATE`) to force the LLM to answer *only* based on provided context, minimizing hallucinations.
    *   **Translation**: Uses the same LLM (`llama3.1`) as a zero-shot translator. It has separate prompt templates for TR→EN and EN→TR translation.
    *   **LCEL (LangChain Expression Language)**: Pipes components together (`prompt | llm | parser`) for cleaner, modern chaining.

### `src/config.py`
*   **Role**: Central configuration.
*   **Details**: Defines model names (`llama3.1:8b`, `nomic-embed-text`), batch sizes, and timeout settings. Allows environment variable overrides for tuning without code changes.

## 4. AI & Algorithms Used

### Embeddings: Nomic Embed Text
We use `nomic-embed-text-v1.5` via Ollama.
*   **Type**: Matryoshka text embedding model.
*   **Why?** It outperforms OpenAI's `text-embedding-ada-002` on several benchmarks while being open weights and runnable locally. It supports long context (8192 tokens), which is crucial for technical manuals.

### Vector Search: MMR (Maximal Marginal Relevance)
Instead of just standard Cosine Similarity, the system is configured to support MMR (though currently standard similarity is active in the main flow, the code supports swapping).
*   **Standard Similarity**: Finds chunks most geometrically close to the query vector.
*   **MMR**: Finds chunks that are similar to the query but *dissimilar to each other*. This provides a diverse set of facts (e.g., one chunk from the introduction, one from the technical specs) rather than 5 variations of the same paragraph.

### LLM: Llama 3.1 8B
*   **Role**: Reasoning, answer synthesis, and translation.
*   **Why?** It's the current state-of-the-art for "small" local models. It has excellent instruction-following capabilities, which is vital for adhering to the "answer only from context" rule.

## 5. Improvement Opportunities

### A. Advanced Retrieval (The "R" in RAG)
1.  **Hybrid Search**: Combine vector search (semantic) with keyword search (BM25).
    *   *Why?* Vectors are bad at exact part numbers (e.g., "Part #99-X-12"). Keywords are great at that. Combining them (Reciprocal Rank Fusion) gives the best of both worlds.
2.  **Re-Ranking (Cross-Encoders)**:
    *   Retrieve 50 chunks quickly (fast/cheap).
    *   Use a Cross-Encoder model (like `bge-reranker`) to score them accurately against the query.
    *   Pass only the top 5 to the LLM.
    *   *Benefit*: Massive accuracy boost.

### B. Intelligent Chunking
1.  **Semantic Chunking**: Instead of splitting by character count, split when the *topic* changes. This uses embeddings to calculate semantic distance between sentences.
2.  **Parent-Child Retrieval**: Embed small chunks (sentences) for accurate search, but return the *full parent paragraph* to the LLM for context.

### C. Conversational Memory
Currently, `ask` treats every question as isolated.
*   **Improvement**: Pass the `chat_history` into the prompt so the user can say "What is the specific gravity?" and then "How do I measure *it*?" (referring back to specific gravity).

### D. User Experience
1.  **Streaming**: Stream the LLM response token-by-token instead of waiting for the full answer.
2.  **Source Highlighting**: If possible, render the PDF page and highlight the specific paragraph used for the answer.

### E. Translation Optimization
*   **Issue**: Using an 8B general LLM for translation is slow.
*   **Fix**: Use a specialized, smaller translation model (e.g., `seamless-m4t` or dedicated NLLB models) or caching (don't re-translate the same query).


