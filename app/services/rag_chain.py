"""
RAG chain: takes a user question, retrieves relevant chunks, and generates
a grounded answer with source citations using an OpenAI chat model.
"""
import logging
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.core.config import settings
from app.services.vector_store import similarity_search

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(
    model=settings.llm_model,
    openai_api_key=settings.openai_api_key,
    temperature=0.1,  # low temperature: favor grounded, consistent answers
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable assistant answering questions using ONLY the "
            "provided context. If the context does not contain the answer, say "
            "you don't have enough information — do not make anything up. "
            "Cite sources inline using the format [source: filename, page].",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ]
)

_chain = _PROMPT | _llm | StrOutputParser()


def _format_context(chunks: List[Document]) -> str:
    parts = []
    for doc in chunks:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[source: {source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, k: int | None = None) -> Dict[str, Any]:
    """Run the full RAG flow: retrieve -> build context -> generate answer."""
    retrieved_chunks = similarity_search(question, k=k)

    if not retrieved_chunks:
        return {
            "answer": "I couldn't find any relevant information in the indexed documents.",
            "sources": [],
        }

    context = _format_context(retrieved_chunks)
    answer = _chain.invoke({"context": context, "question": question})

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
            "chunk_id": doc.metadata.get("chunk_id"),
            "snippet": doc.page_content[:200],
        }
        for doc in retrieved_chunks
    ]

    return {"answer": answer, "sources": sources}
