"""
Document ingestion: loads PDFs, splits into chunks ready for embedding.
"""
import os
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.core.config import settings

logger = logging.getLogger(__name__)

# A page with fewer than this many characters is treated as "no real text"
# and gets OCR'd instead (covers scanned pages and near-empty pages alike).
MIN_CHARS_PER_PAGE = 20


def _ocr_pdf(file_path: str, source_name: str) -> List[Document]:
    """Fallback for scanned/image-based PDFs: rasterize each page and run OCR."""
    from pdf2image import convert_from_path
    import pytesseract

    logger.info("Running OCR on %s (no extractable text found)", source_name)
    images = convert_from_path(file_path)

    documents = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        documents.append(
            Document(
                page_content=text,
                metadata={"source": source_name, "page": i},
            )
        )
    return documents


def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF and return one LangChain Document per page.

    Tries normal text extraction first; if that yields effectively no text
    (e.g. a scanned/image-based PDF), falls back to OCR automatically.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    source_name = os.path.basename(file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = source_name

    total_chars = sum(len(doc.page_content.strip()) for doc in documents)
    avg_chars_per_page = total_chars / max(len(documents), 1)

    if avg_chars_per_page < MIN_CHARS_PER_PAGE:
        documents = _ocr_pdf(file_path, source_name)

    logger.info("Loaded %d pages from %s", len(documents), source_name)
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks sized for embedding + context window."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Give each chunk a stable id (source + running index) for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'unknown')}_{i}"

    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def process_pdf(file_path: str) -> List[Document]:
    """Full ingestion pipeline for one PDF: load -> chunk."""
    raw_docs = load_pdf(file_path)
    return chunk_documents(raw_docs)
