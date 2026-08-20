# AI Knowledge Assistant (RAG)

Context-aware question answering over PDF documents, built with **FastAPI**, **LangChain**,
the **OpenAI API**, and **FAISS**.

## Demo
![AI Knowledge Assistant Demo] (AI_Knowledge_Image.png)


## How it works

```
PDF upload → text extraction → chunking → OpenAI embeddings → FAISS index
                                                                    │
User question ─────────────────────────────────────────► similarity search
                                                                    │
                                                    top-k relevant chunks
                                                                    │
                                                        LangChain prompt + LLM
                                                                    │
                                                        Answer + cited sources
```

## Project structure

```
rag-assistant/
├── app/
│   ├── main.py              # FastAPI app entrypoint
│   ├── core/
│   │   └── config.py         # Settings (env vars)
│   ├── api/
│   │   ├── routes.py         # /ingest, /query, /health endpoints
│   │   └── schemas.py        # Pydantic request/response models
│   └── services/
│       ├── ingestion.py      # PDF loading + chunking
│       ├── vector_store.py   # FAISS index management
│       └── rag_chain.py      # Retrieval + prompt + LLM generation
├── data/
│   ├── uploads/               # Uploaded PDFs land here
│   └── vector_store/          # Persisted FAISS index
├── requirements.txt
└── .env.example
```

## Setup

1. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and add your `OPENAI_API_KEY`.

3. **Run the server**

   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be live at `http://localhost:8000`, with interactive docs at
   `http://localhost:8000/docs`.

## Usage

### 1. Ingest a document

```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
  -F "file=@/path/to/your/document.pdf"
```

Response:
```json
{
  "filename": "document.pdf",
  "chunks_indexed": 42,
  "message": "Document ingested successfully."
}
```

### 2. Ask a question

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the termination clause in this contract?"}'
```

Response:
```json
{
  "answer": "According to the document, either party may terminate... [source: document.pdf, page 4]",
  "sources": [
    {
      "source": "document.pdf",
      "page": 4,
      "chunk_id": "document.pdf_12",
      "snippet": "Either party may terminate this agreement with 30 days written notice..."
    }
  ]
}
```

### 3. Health check

```bash
curl "http://localhost:8000/api/v1/health"
```

## Configuration reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model for generation |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `VECTOR_STORE_PATH` | `data/vector_store` | Where the FAISS index is persisted |
| `UPLOAD_DIR` | `data/uploads` | Where uploaded PDFs are stored |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks retrieved per query |

## Design notes / things to know

- **FAISS persistence**: the index is saved to disk after every ingestion (`save_local`),
  and loaded lazily on first query or ingestion (`load_local`). For production, consider
  loading it once at app startup instead.
- **`allow_dangerous_deserialization=True`**: required by LangChain's FAISS loader.
  This is safe here because the app only ever loads an index it created itself — never
  load a FAISS index from an untrusted source with this flag.
- **Chunking strategy**: uses `RecursiveCharacterTextSplitter`, which tries to split on
  paragraph → sentence → word boundaries in that order. Tune `CHUNK_SIZE`/`CHUNK_OVERLAP`
  based on your documents (denser technical docs often want smaller chunks).
- **Grounding**: the system prompt explicitly instructs the model to say when it doesn't
  know, rather than hallucinate — reducing (not eliminating) unsupported answers.
- **Scaling beyond FAISS**: this uses local FAISS, which is fine up to roughly hundreds
  of thousands of vectors. Beyond that, or for multi-instance deployments, migrate to a
  managed vector DB (Pinecone, Weaviate, pgvector).

## Not included (see project estimate for add-on scope)

- Authentication / API keys for your own API
- Deployment config (Docker, CI/CD)
- Streaming responses
- Multi-tenant document isolation
- Frontend UI
- Support for non-PDF formats (easy to add — swap `PyPDFLoader` for
  `Docx2txtLoader`, `UnstructuredHTMLLoader`, etc.)
