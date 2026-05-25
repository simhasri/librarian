"""
ingestion/chunker.py

Chunks raw book text and enriches each chunk with context
using Claude — this is Anthropic's Contextual Retrieval technique.

The problem we're solving:
    A naive chunk might say "he felt the walls closing in" with no
    indication of who, what book, or what situation. When we search
    for "imprisonment themes in Victorian literature" this chunk might
    get missed entirely because the embedding has no context.

The fix:
    Before embedding each chunk, we ask Claude to write 1-2 sentences
    explaining where this chunk fits in the broader document. We prepend
    that to the chunk before embedding. Now the embedding is rich with
    context and retrieval accuracy improves significantly.
"""

import json
import time
from pathlib import Path
from typing import Generator
import anthropic
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 400       # words per chunk
CHUNK_OVERLAP = 50     # words overlap between chunks

# Our book metadata — matches the downloader
BOOK_METADATA = {
    "98_A_Tale_of_Two_Cities.txt":          {"author": "Charles Dickens",  "genre": "fiction", "era": "19th century"},
    "1342_Pride_and_Prejudice.txt":         {"author": "Jane Austen",      "genre": "fiction", "era": "19th century"},
    "2701_Moby_Dick.txt":                   {"author": "Herman Melville",  "genre": "fiction", "era": "19th century"},
    "84_Frankenstein.txt":                  {"author": "Mary Shelley",     "genre": "gothic",  "era": "19th century"},
    "174_The_Picture_of_Dorian_Gray.txt":   {"author": "Oscar Wilde",      "genre": "gothic",  "era": "19th century"},
    "11_Alice_in_Wonderland.txt":           {"author": "Lewis Carroll",    "genre": "fantasy", "era": "19th century"},
    "1952_The_Yellow_Wallpaper.txt":        {"author": "Charlotte Gilman", "genre": "fiction", "era": "19th century"},
    "76_Huckleberry_Finn.txt":              {"author": "Mark Twain",       "genre": "fiction", "era": "19th century"},
    "1080_A_Modest_Proposal.txt":           {"author": "Jonathan Swift",   "genre": "satire",  "era": "18th century"},
    "2600_War_and_Peace.txt":               {"author": "Leo Tolstoy",      "genre": "fiction", "era": "19th century"},
}

client = anthropic.Anthropic()


def split_into_chunks(text: str) -> list[str]:
    """
    Splits text into overlapping word-based chunks.

    Why overlapping? If a key sentence sits right at the boundary
    between two chunks, overlap ensures it appears in at least one
    chunk fully — so it won't get missed during retrieval.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def enrich_chunk_with_context(
    chunk: str,
    full_text: str,
    title: str,
    author: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """
    The core of Contextual Retrieval.

    Sends the full document + the specific chunk to Claude and asks
    it to write a short contextual summary. We prepend this to the
    chunk before embedding.

    We only send the first 3000 words of the full text to Claude
    to keep costs low — enough context for Claude to understand
    the document without sending the entire book every time.
    """
    # Truncate full text to keep API costs low
    doc_preview = " ".join(full_text.split()[:3000])

    prompt = f"""Here is a book excerpt for context:
<document>
Title: {title} by {author}
{doc_preview}
</document>

Here is a specific chunk from this book (chunk {chunk_index + 1} of {total_chunks}):
<chunk>
{chunk}
</chunk>

Write 1-2 sentences that explain where this chunk fits in the broader document.
Include the book title, author, and any relevant plot or thematic context.
Be specific and concise. Do not summarize the chunk itself — just provide context for it.
Reply with only the contextual sentences, nothing else."""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Haiku is fast and cheap for this task
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    context = response.content[0].text.strip()

    # Prepend context to chunk — this is what gets embedded
    return f"{context}\n\n{chunk}"


def process_book(file_path: Path) -> list[dict]:
    """
    Full pipeline for one book:
    1. Read raw text
    2. Split into chunks
    3. Enrich each chunk with Claude context
    4. Return list of enriched chunk dicts with metadata
    """
    filename = file_path.name
    metadata = BOOK_METADATA.get(filename, {})
    title = filename.replace(".txt", "").replace("_", " ").split(" ", 1)[1]
    author = metadata.get("author", "Unknown")
    genre = metadata.get("genre", "unknown")
    era = metadata.get("era", "unknown")

    print(f"\n📖 Processing: {title} by {author}")

    full_text = file_path.read_text(encoding="utf-8")
    chunks = split_into_chunks(full_text)
    print(f"   → {len(chunks)} chunks to enrich")

    enriched_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"   Enriching chunk {i + 1}/{len(chunks)}...", end="\r")

        try:
            enriched = enrich_chunk_with_context(
                chunk, full_text, title, author, i, len(chunks)
            )
        except Exception as e:
            print(f"\n   ⚠️  Failed to enrich chunk {i + 1}: {e}")
            enriched = chunk  # fall back to raw chunk

        enriched_chunks.append({
            "text": enriched,
            "raw_text": chunk,
            "source": filename,
            "title": title,
            "author": author,
            "genre": genre,
            "era": era,
            "chunk_index": i,
            "total_chunks": len(chunks),
        })

        # Small delay to avoid hitting API rate limits
        time.sleep(0.3)

    print(f"   ✅ Done — {len(enriched_chunks)} chunks enriched")
    return enriched_chunks


def process_all_books():
    """
    Processes all downloaded books and saves enriched chunks
    as JSON files in data/processed/
    """
    book_files = list(RAW_DIR.glob("*.txt"))

    if not book_files:
        print("⚠️  No books found in data/raw/ — run downloader.py first")
        return

    print(f"🚀 Processing {len(book_files)} books with contextual enrichment...\n")
    print("⚠️  This will make Claude API calls for each chunk.")
    print("    Haiku is used to keep costs minimal.\n")

    all_chunks = []

    for book_path in book_files:
        chunks = process_book(book_path)
        all_chunks.extend(chunks)

        # Save per-book file as we go
        # so if it crashes midway you don't lose everything
        book_output = PROCESSED_DIR / f"{book_path.stem}_chunks.json"
        with open(book_output, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All done — {len(all_chunks)} total chunks processed")
    print(f"   Saved to data/processed/")


if __name__ == "__main__":
    process_all_books()