"""
Vector store management: builds, persists, loads, and queries a FAISS index.
"""
import os
import logging
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import settings

logger = logging.getLogger(__name__)

_embeddings = OpenAIEmbeddings(
    model=settings.embedding_model,
    openai_api_key=settings.openai_api_key,
)

_vector_store: Optional[FAISS] = None


def _index_files_exist() -> bool:
    return os.path.exists(os.path.join(settings.vector_store_path, "index.faiss"))


def get_vector_store() -> FAISS:
    """Return the in-memory FAISS store, loading it from disk if needed."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    if _index_files_exist():
        logger.info("Loading FAISS index from %s", settings.vector_store_path)
        _vector_store = FAISS.load_local(
            settings.vector_store_path,
            _embeddings,
            allow_dangerous_deserialization=True,  # safe: we only load our own index
        )
        return _vector_store

    raise RuntimeError(
        "No vector store found. Ingest at least one document before querying."
    )


def add_documents(chunks: List[Document]) -> int:
    """Embed chunks and add them to the FAISS index, creating it if it doesn't exist yet."""
    global _vector_store

    if _vector_store is None and _index_files_exist():
        _vector_store = FAISS.load_local(
            settings.vector_store_path, _embeddings, allow_dangerous_deserialization=True
        )

    if _vector_store is None:
        logger.info("Creating new FAISS index with %d chunks", len(chunks))
        _vector_store = FAISS.from_documents(chunks, _embeddings)
    else:
        logger.info("Adding %d chunks to existing FAISS index", len(chunks))
        _vector_store.add_documents(chunks)

    os.makedirs(settings.vector_store_path, exist_ok=True)
    _vector_store.save_local(settings.vector_store_path)
    return len(chunks)


def similarity_search(query: str, k: Optional[int] = None) -> List[Document]:
    """Retrieve the top-k most relevant chunks for a query."""
    store = get_vector_store()
    k = k or settings.retrieval_top_k
    return store.similarity_search(query, k=k)
