"""
FastAPI application entrypoint for the AI Knowledge Assistant (RAG).
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI Knowledge Assistant (RAG)",
    description="Context-aware Q&A over PDF documents using LangChain, OpenAI, and FAISS.",
    version="1.0.0",
)

# Adjust origins for your actual frontend in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {"message": "AI Knowledge Assistant API is running. See /docs for usage."}
