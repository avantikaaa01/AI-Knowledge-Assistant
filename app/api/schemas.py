from typing import List, Optional
from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str = "Document ingested successfully."


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override retrieval depth")


class SourceChunk(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str = "ok"
