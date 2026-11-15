"""
Vector store service.
Manages local FAISS index creation, persistence, chunk indexing, and retriever creation
using HuggingFaceEmbeddings (all-MiniLM-L6-v2).
"""
import logging
from pathlib import Path
from typing import List, Optional, Union

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global singletons / caches
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[FAISS] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes and caches the HuggingFaceEmbeddings model singleton.
    Uses 'sentence-transformers/all-MiniLM-L6-v2'.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Initializing HuggingFaceEmbeddings with model '%s'", settings.EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings


def index_chunks(
    chunks: Union[List[Document], List[str]],
    persist_dir: Optional[Union[str, Path]] = None
) -> FAISS:
    """
    Indexes text chunks (Documents or strings) into a local FAISS vector store.
    If an existing index exists at persist_dir, new documents are merged in;
    otherwise, a new index is created and saved.

    Args:
        chunks: List of LangChain Document objects or raw text strings.
        persist_dir: Directory where FAISS index files should be persisted.
                     Defaults to settings.FAISS_INDEX_DIR.

    Returns:
        FAISS: The updated or newly initialized FAISS vector store instance.
    """
    global _vector_store
    target_dir = Path(persist_dir) if persist_dir else settings.FAISS_INDEX_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embedding_model()

    # Convert strings to Documents if necessary
    doc_chunks: List[Document] = [
        chunk if isinstance(chunk, Document) else Document(page_content=chunk)
        for chunk in chunks
    ]

    index_file = target_dir / "index.faiss"

    if index_file.exists():
        logger.info("Existing FAISS index found at '%s'. Merging new chunks...", target_dir)
        try:
            if _vector_store is None:
                _vector_store = FAISS.load_local(
                    folder_path=str(target_dir),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
            _vector_store.add_documents(doc_chunks)
        except Exception as exc:
            logger.warning("Failed to load existing index (%s), creating fresh one.", exc)
            _vector_store = FAISS.from_documents(doc_chunks, embeddings)
    else:
        logger.info("Creating new FAISS vector index with %d chunks...", len(doc_chunks))
        _vector_store = FAISS.from_documents(doc_chunks, embeddings)

    # Persist locally to disk
    _vector_store.save_local(folder_path=str(target_dir))
    logger.info("FAISS vector store saved to '%s'", target_dir)

    return _vector_store


def get_retriever(
    persist_dir: Optional[Union[str, Path]] = None,
    k: int = 4
) -> VectorStoreRetriever:
    """
    Loads and returns a LangChain VectorStoreRetriever from the FAISS vector store.

    Args:
        persist_dir: Directory where FAISS index is saved.
        k: Number of nearest document chunks to retrieve per query.

    Returns:
        VectorStoreRetriever: Configured retriever ready for RAG pipelines.

    Raises:
        FileNotFoundError: If no FAISS index exists on disk and none is loaded in memory.
    """
    global _vector_store
    target_dir = Path(persist_dir) if persist_dir else settings.FAISS_INDEX_DIR
    index_file = target_dir / "index.faiss"

    if _vector_store is None:
        if not index_file.exists():
            raise FileNotFoundError(
                f"No FAISS index found at '{target_dir}'. "
                "Please upload a document to initialize the vector store before querying."
            )
        embeddings = get_embedding_model()
        logger.info("Loading FAISS index from '%s'...", target_dir)
        _vector_store = FAISS.load_local(
            folder_path=str(target_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

    return _vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def is_index_ready(persist_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    Checks if a valid vector store index is available either in memory or on disk.
    """
    global _vector_store
    if _vector_store is not None:
        return True
    target_dir = Path(persist_dir) if persist_dir else settings.FAISS_INDEX_DIR
    return (target_dir / "index.faiss").exists()

