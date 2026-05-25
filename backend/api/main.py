"""
api/main.py

FastAPI backend that wires together retrieval + generation
into clean endpoints the frontend can call.

Endpoints:
    POST /search   → hybrid retrieval only (no LLM)
    POST /ask      → full RAG pipeline with faithfulness check
    GET  /health   → sanity check
    GET  /stats    → info about the indexed collection
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import time

from backend.retrieval.retriever import Retriever
from backend.generation.generator import ask

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Librarian API",
    description="Semantic search and Q&A over classic literature",
    version="1.0.0",
)

# CORS — allows your Next.js frontend to talk to this API
# In production you'd lock this down to your Vercel domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever once at startup — expensive to reload every request
retriever = Retriever()


# ── Request/Response models ──────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 6
    genre: Optional[str] = None
    era: Optional[str] = None
    author: Optional[str] = None


class AskRequest(BaseModel):
    query: str
    top_k: int = 6
    genre: Optional[str] = None
    era: Optional[str] = None
    author: Optional[str] = None


class SearchResult(BaseModel):
    title: str
    author: str
    genre: str
    era: str
    raw_text: str
    rerank_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    latency_ms: float


class FaithfulnessResult(BaseModel):
    score: float
    label: str
    reasoning: str


class AskResponse(BaseModel):
    query: str
    answer: str
    faithfulness: FaithfulnessResult
    sources: list[dict]
    results: list[SearchResult]
    latency_ms: float


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick sanity check — used by deployment platforms to verify the app is running."""
    return {"status": "ok", "message": "Librarian API is running"}


@app.get("/stats")
def stats():
    """Returns info about the indexed collection."""
    return {
        "total_chunks": retriever.collection.count(),
        "collection": "librarian",
        "books": [
            "A Modest Proposal",
            "Alice in Wonderland",
            "Pride and Prejudice",
            "The Picture of Dorian Gray",
        ]
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Hybrid retrieval endpoint.
    Returns ranked passages — no LLM involved, very fast.
    Use this to power the search results shelf in the UI.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.time()

    results = retriever.search(
        query=request.query,
        top_k=request.top_k,
        genre_filter=request.genre,
        era_filter=request.era,
        author_filter=request.author,
    )

    latency = round((time.time() - start) * 1000, 2)

    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
        latency_ms=latency,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Full RAG pipeline endpoint.
    1. Retrieves relevant passages
    2. Generates a grounded answer with Claude
    3. Scores faithfulness of the answer
    Returns everything including the faithfulness badge data.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.time()

    # Step 1: Retrieve
    results = retriever.search(
        query=request.query,
        top_k=request.top_k,
        genre_filter=request.genre,
        era_filter=request.era,
        author_filter=request.author,
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant passages found for this query"
        )

    # Step 2: Generate + faithfulness check
    generation = ask(request.query, results)

    latency = round((time.time() - start) * 1000, 2)

    return AskResponse(
        query=request.query,
        answer=generation["answer"],
        faithfulness=FaithfulnessResult(**generation["faithfulness"]),
        sources=generation["sources"],
        results=[SearchResult(**r) for r in results],
        latency_ms=latency,
    )