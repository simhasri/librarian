"""
retrieval/retriever.py

Hybrid search: BM25 (keyword) + ChromaDB (semantic vector search)
combined and reranked with a cross-encoder.

Why hybrid?
    Pure vector search is great for conceptual queries like
    "themes of isolation" but misses exact terms like character
    names or specific years. BM25 catches those. Combining both
    gives you the best of both worlds.

Why reranking?
    Both searches return candidates, not final answers. The
    cross-encoder reranker reads query + each candidate together
    and scores how relevant they actually are — much more accurate
    than either search alone.
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "librarian"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_VECTOR = 20    # candidates from vector search
TOP_K_BM25 = 20      # candidates from BM25
TOP_K_FINAL = 6      # final results after reranking


class Retriever:
    def __init__(self):
        print("🔍 Initializing retriever...")

        # Embedding model for query encoding
        self.embedder = SentenceTransformer(EMBED_MODEL)

        # Cross-encoder for reranking
        print("   Loading reranker...")
        self.reranker = CrossEncoder(RERANK_MODEL)

        # ChromaDB connection
        client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection(COLLECTION_NAME)

        # Load all documents for BM25
        # BM25 needs all docs in memory — fine for our scale
        print("   Building BM25 index...")
        self._build_bm25_index()

        print(f"✅ Retriever ready — {self.collection.count()} chunks indexed\n")

    def _build_bm25_index(self):
        """
        Fetches all documents from ChromaDB and builds a BM25 index.

        BM25 works on tokenized words, so we split each document
        into a list of lowercase words before indexing.
        """
        results = self.collection.get(include=["documents", "metadatas"])
        self.all_docs = results["documents"]
        self.all_metadatas = results["metadatas"]
        self.all_ids = results["ids"]

        # Tokenize for BM25
        tokenized = [doc.lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized)

    def _vector_search(self, query: str, top_k: int = TOP_K_VECTOR) -> list[dict]:
        """Semantic search using ChromaDB + embeddings."""
        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            candidates.append({
                "text": doc,
                "metadata": meta,
                "vector_score": round(1 - dist, 4),
                "source": "vector",
            })
        return candidates

    def _bm25_search(self, query: str, top_k: int = TOP_K_BM25) -> list[dict]:
        """Keyword search using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top K indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        candidates = []
        for idx in top_indices:
            if scores[idx] > 0:  # only include if there's actual keyword match
                candidates.append({
                    "text": self.all_docs[idx],
                    "metadata": self.all_metadatas[idx],
                    "bm25_score": round(float(scores[idx]), 4),
                    "source": "bm25",
                })
        return candidates

    def _rerank(self, query: str, candidates: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
        """
        Reranks candidates using a cross-encoder.

        Unlike bi-encoders (which encode query and doc separately),
        a cross-encoder reads query + document together — much more
        accurate but slower, which is why we only rerank the top
        candidates, not the full corpus.
        """
        if not candidates:
            return []

        # Deduplicate by text first
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c["text"] not in seen:
                seen.add(c["text"])
                unique_candidates.append(c)

        # Score each candidate
        pairs = [[query, c["text"]] for c in unique_candidates]
        scores = self.reranker.predict(pairs)

        # Attach rerank scores
        for candidate, score in zip(unique_candidates, scores):
            candidate["rerank_score"] = round(float(score), 4)

        # Sort by rerank score and return top K
        reranked = sorted(unique_candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def search(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        genre_filter: Optional[str] = None,
        era_filter: Optional[str] = None,
        author_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Main search method — runs hybrid search + reranking.

        Args:
            query: The user's search query
            top_k: Number of final results to return
            genre_filter: Optional filter e.g. "gothic"
            era_filter: Optional filter e.g. "19th century"
            author_filter: Optional filter e.g. "Jane Austen"

        Returns:
            List of ranked results with text, metadata, and scores
        """
        # Run both searches in parallel (conceptually)
        vector_results = self._vector_search(query, TOP_K_VECTOR)
        bm25_results = self._bm25_search(query, TOP_K_BM25)

        # Merge candidates
        all_candidates = vector_results + bm25_results

        # Apply metadata filters if provided
        if genre_filter:
            all_candidates = [c for c in all_candidates if c["metadata"].get("genre") == genre_filter]
        if era_filter:
            all_candidates = [c for c in all_candidates if c["metadata"].get("era") == era_filter]
        if author_filter:
            all_candidates = [c for c in all_candidates if c["metadata"].get("author") == author_filter]

        # Rerank merged candidates
        final_results = self._rerank(query, all_candidates, top_k)

        # Format for output
        formatted = []
        for r in final_results:
            formatted.append({
                "text": r["text"],
                "raw_text": r["metadata"].get("raw_text", r["text"]),
                "title": r["metadata"].get("title", ""),
                "author": r["metadata"].get("author", ""),
                "genre": r["metadata"].get("genre", ""),
                "era": r["metadata"].get("era", ""),
                "chunk_index": r["metadata"].get("chunk_index", 0),
                "rerank_score": r["rerank_score"],
            })

        return formatted


if __name__ == "__main__":
    retriever = Retriever()

    # Test query
    query = "feeling trapped by society's expectations"
    print(f"Query: '{query}'\n")
    results = retriever.search(query)

    for i, r in enumerate(results, 1):
        print(f"Result {i}: {r['title']} by {r['author']}")
        print(f"  Score: {r['rerank_score']}")
        print(f"  {r['raw_text'][:150]}...")
        print()