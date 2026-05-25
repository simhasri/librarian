"""
ingestion/downloader.py

Downloads books from Project Gutenberg and saves them
to data/raw/ as plain text files.
"""

import requests
import time
from pathlib import Path

# Where we save downloaded books
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Our starter library — (book_id, title, author, genre, era)
BOOKS = [
    (98,   "A Tale of Two Cities",              "Charles Dickens",  "fiction", "19th century"),
    (1342, "Pride and Prejudice",               "Jane Austen",      "fiction", "19th century"),
    (2701, "Moby Dick",                         "Herman Melville",  "fiction", "19th century"),
    (84,   "Frankenstein",                      "Mary Shelley",     "gothic",  "19th century"),
    (174,  "The Picture of Dorian Gray",        "Oscar Wilde",      "gothic",  "19th century"),
    (11,   "Alice in Wonderland",               "Lewis Carroll",    "fantasy", "19th century"),
    (1952, "The Yellow Wallpaper",              "Charlotte Gilman", "fiction", "19th century"),
    (76,   "Huckleberry Finn",                  "Mark Twain",       "fiction", "19th century"),
    (1080, "A Modest Proposal",                 "Jonathan Swift",   "satire",  "18th century"),
    (2600, "War and Peace",                     "Leo Tolstoy",      "fiction", "19th century"),
]


def download_book(book_id: int, title: str) -> str | None:
    """
    Downloads a single book from Gutenberg.
    Tries two URL formats since Gutenberg isn't always consistent.
    Returns the text if successful, None if it fails.
    """
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]

    for url in urls:
        try:
            print(f"  Trying: {url}")
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                print(f"  ✅ Downloaded: {title}")
                return response.text
        except requests.RequestException as e:
            print(f"  ❌ Failed: {e}")
            continue

    print(f"  ⚠️  Could not download: {title}")
    return None


def clean_gutenberg_text(text: str) -> str:
    """
    Gutenberg books have a legal header and footer we don't want
    in our search results. This strips them out.

    Why this matters: if we left the header in, searches about
    'copyright' or 'Project Gutenberg' would return garbage results.
    """
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = text.find("\n", idx) + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def download_all():
    """
    Downloads all books in our BOOKS list,
    saves each as a .txt file in data/raw/
    """
    print(f"📚 Downloading {len(BOOKS)} books from Project Gutenberg...\n")
    success = 0

    for book_id, title, author, genre, era in BOOKS:
        print(f"📖 {title} — {author}")

        # Skip if already downloaded
        output_path = RAW_DIR / f"{book_id}_{title.replace(' ', '_')}.txt"
        if output_path.exists():
            print(f"  ⏭️  Already exists, skipping\n")
            success += 1
            continue

        text = download_book(book_id, title)
        if text:
            cleaned = clean_gutenberg_text(text)
            output_path.write_text(cleaned, encoding="utf-8")
            success += 1

        # Be polite to Gutenberg's servers
        time.sleep(1)
        print()

    print(f"✅ Done — {success}/{len(BOOKS)} books downloaded to data/raw/")


if __name__ == "__main__":
    download_all()