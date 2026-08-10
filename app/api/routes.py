"""
API routes: document upload/ingestion and question-answering endpoints.
"""
import os
import logging
import shutil

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.core.config import settings
from app.services.ingestion import process_pdf
from app.services.vector_store import add_documents
from app.services.rag_chain import answer_question
from app.api.schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload a PDF, chunk it, embed it, and add it to the FAISS index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = process_pdf(file_path)
        if not chunks:
            raise HTTPException(
                status_code=422, detail="No extractable text found in this PDF."
            )

        num_indexed = add_documents(chunks)

        return IngestResponse(filename=file.filename, chunks_indexed=num_indexed)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question and get a grounded answer with source citations."""
    try:
        result = answer_question(request.question, k=request.top_k)
        return QueryResponse(**result)
    except RuntimeError as exc:
        # e.g. no documents ingested yet
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
