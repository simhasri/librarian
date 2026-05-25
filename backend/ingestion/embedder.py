"""
ingestion/embedder.py

Reads enriched chunks from data/processed/,
embeds them using sentence-transformers,
and stores them in ChromaDB with metadata.

Why ChromaDB?
    It's a local vector database — no server needed, no cost,
    and it persists to disk so we don't re-embed every time.

Why all-MiniLM-L6-v2?
    Fast, small, and punches above its weight for semantic search.
    Good default for a project like this.
"""

import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path("data/processed")
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "librarian"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_all_chunks() -> list[dict]:
    """Loads all processed chunk JSON files from data/processed/"""
    all_chunks = []
    json_files = list(PROCESSED_DIR.glob("*_chunks.json"))

    if not json_files:
        print("⚠️  No processed chunks found — run chunker.py first")
        return []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)
        print(f"  📄 Loaded {len(chunks)} chunks from {json_file.name}")

    return all_chunks


def embed_and_store():
    """
    Main function:
    1. Load all enriched chunks
    2. Embed them in batches
    3. Store in ChromaDB with metadata
    """
    print("🔍 Loading chunks...")
    chunks = load_all_chunks()

    if not chunks:
        return

    print(f"\n✅ Total chunks to embed: {len(chunks)}")

    # Load embedding model
    print(f"\n📦 Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Init ChromaDB
    print(f"💾 Initializing ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

    # Fresh collection each time we embed
    # In production you'd do incremental updates instead
    try:
        client.delete_collection(COLLECTION_NAME)
        print("   Cleared existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed in batches of 64 for efficiency
    BATCH_SIZE = 64
    total_stored = 0

    print(f"\n🚀 Embedding and storing {len(chunks)} chunks in batches of {BATCH_SIZE}...\n")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]

        # Extract text to embed — this is the contextually enriched text
        texts = [c["text"] for c in batch]

        # Generate embeddings
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        # Build unique IDs and metadata for ChromaDB
        ids = [f"{c['source']}__chunk_{c['chunk_index']}" for c in batch]
        metadatas = [
            {
                "source":      c["source"],
                "title":       c["title"],
                "author":      c["author"],
                "genre":       c["genre"],
                "era":         c["era"],
                "chunk_index": c["chunk_index"],
                "raw_text":    c["raw_text"],  # store raw text separately for display
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_stored += len(batch)
        print(f"  Stored {total_stored}/{len(chunks)} chunks...", end="\r")

    print(f"\n\n✅ Done — {total_stored} chunks embedded and stored in ChromaDB")
    print(f"   Collection: '{COLLECTION_NAME}'")
    print(f"   Location: {CHROMA_DIR}/")


if __name__ == "__main__":
    embed_and_store()