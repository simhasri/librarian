"""
generation/generator.py

Two things happen here:
1. Claude generates an answer grounded in retrieved chunks
2. A second Claude call scores whether the answer is faithful
   to the retrieved context (faithfulness check)

Why faithfulness checking?
    RAG systems can still hallucinate — Claude might "know" the
    answer from training data and ignore the retrieved context.
    The faithfulness checker catches this by asking a second
    Claude call: "is every claim in this answer supported by
    the context?" and returns a 0-1 score.

    This is what separates a serious RAG project from a tutorial.
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-haiku-4-5-20251001"  # fast and cheap for both calls

SYSTEM_PROMPT = """You are Librarian, an expert literary assistant with deep knowledge
of classic literature. You help users explore themes, characters, and ideas across books.

When answering:
- Ground every claim in the provided context passages
- Cite which book and author each point comes from
- Be analytical and insightful, not just descriptive
- If the context doesn't contain enough information, say so honestly
- Keep answers focused — 3 to 5 sentences unless more detail is needed"""


def format_context(results: list[dict]) -> str:
    """
    Formats retrieved chunks into a readable context block
    for the Claude prompt.
    """
    if not results:
        return "No relevant passages found."

    sections = []
    for i, r in enumerate(results, 1):
        sections.append(
            f"[Passage {i} — {r['title']} by {r['author']}]\n{r['raw_text']}"
        )
    return "\n\n---\n\n".join(sections)


def generate_answer(query: str, results: list[dict]) -> str:
    """
    Sends retrieved context + user query to Claude
    and returns a grounded answer.
    """
    context = format_context(results)

    prompt = f"""Here are relevant passages from the literature database:

{context}

---

User question: {query}

Answer based on the passages above. Cite the specific books you draw from."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text.strip()


def check_faithfulness(answer: str, results: list[dict]) -> dict:
    """
    Faithfulness checker — the key differentiator of this project.

    Asks Claude to score whether every claim in the answer
    is supported by the retrieved context.

    Returns:
        {
            "score": float (0.0 to 1.0),
            "label": "high" | "medium" | "low",
            "reasoning": str
        }

    Score guide:
        0.8 - 1.0 → high   (green badge in UI)
        0.5 - 0.8 → medium (yellow badge)
        0.0 - 0.5 → low    (red badge — answer may be hallucinated)
    """
    context = format_context(results)

    prompt = f"""You are a faithfulness evaluator for a RAG system.

Retrieved context:
{context}

---

Generated answer:
{answer}

---

Evaluate whether every claim in the answer is supported by the retrieved context.

Respond in this exact format:
SCORE: [a number between 0.0 and 1.0]
LABEL: [high, medium, or low]
REASONING: [one sentence explaining your score]

Rules:
- 0.8 to 1.0 = all claims clearly supported by context
- 0.5 to 0.8 = most claims supported, minor extrapolation
- 0.0 to 0.5 = significant claims not found in context"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Parse the structured response
    score = 0.5
    label = "medium"
    reasoning = "Could not parse faithfulness response."

    for line in raw.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("LABEL:"):
            label = line.replace("LABEL:", "").strip().lower()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {
        "score": score,
        "label": label,
        "reasoning": reasoning,
    }


def ask(query: str, results: list[dict]) -> dict:
    """
    Full generation pipeline:
    1. Generate answer from retrieved context
    2. Check faithfulness of that answer
    3. Return everything together

    Returns:
        {
            "answer": str,
            "faithfulness": {"score": float, "label": str, "reasoning": str},
            "sources": [{"title": str, "author": str}, ...]
        }
    """
    answer = generate_answer(query, results)
    faithfulness = check_faithfulness(answer, results)

    sources = list({
        (r["title"], r["author"]) for r in results
    })
    sources = [{"title": t, "author": a} for t, a in sources]

    return {
        "answer": answer,
        "faithfulness": faithfulness,
        "sources": sources,
    }


if __name__ == "__main__":
    # Quick test without needing the full retriever
    mock_results = [
        {
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "raw_text": "It is a truth universally acknowledged, that a single man in possession "
                        "of a good fortune, must be in want of a wife. However little known the "
                        "feelings or views of such a man may be on his first entering a "
                        "neighbourhood, this truth is so well fixed in the minds of the "
                        "surrounding families, that he is considered as the rightful property "
                        "of some one or other of their daughters.",
        }
    ]

    query = "How does Austen portray the social pressure of marriage?"
    print(f"Query: {query}\n")

    result = ask(query, mock_results)

    print(f"Answer:\n{result['answer']}\n")
    print(f"Faithfulness: {result['faithfulness']['label'].upper()} "
          f"({result['faithfulness']['score']})")
    print(f"Reasoning: {result['faithfulness']['reasoning']}")