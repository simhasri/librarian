"use client";

import { useState } from "react";

const API = "http://127.0.0.1:8000";

type Result = {
  title: string;
  author: string;
  genre: string;
  era: string;
  raw_text: string;
  rerank_score: number;
};

type Faithfulness = {
  score: number;
  label: string;
  reasoning: string;
};

type AskResponse = {
  query: string;
  answer: string;
  faithfulness: Faithfulness;
  sources: { title: string; author: string }[];
  results: Result[];
  latency_ms: number;
};

function FaithfulnessBadge({ faithfulness }: { faithfulness: Faithfulness }) {
  const colors = {
    high:   "bg-green-900 text-green-300 border-green-700",
    medium: "bg-yellow-900 text-yellow-300 border-yellow-700",
    low:    "bg-red-900 text-red-300 border-red-700",
  };
  const color = colors[faithfulness.label as keyof typeof colors] ?? colors.medium;

  return (
    <div className={`border rounded-lg p-3 text-sm ${color}`}>
      <div className="font-semibold mb-1">
        Faithfulness: {faithfulness.label.toUpperCase()} ({faithfulness.score.toFixed(2)})
      </div>
      <div className="opacity-80">{faithfulness.reasoning}</div>
    </div>
  );
}

function ResultCard({ result, index }: { result: Result; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const preview = result.raw_text.slice(0, 200);

  return (
    <div className="border border-stone-700 rounded-lg p-4 bg-stone-900 hover:border-amber-700 transition-colors">
      <div className="flex justify-between items-start mb-2">
        <div>
          <span className="text-amber-500 text-xs font-mono mr-2">#{index + 1}</span>
          <span className="text-white font-semibold">{result.title}</span>
          <span className="text-stone-400 text-sm ml-2">— {result.author}</span>
        </div>
        <div className="flex gap-2 text-xs">
          <span className="bg-stone-800 text-stone-400 px-2 py-1 rounded">{result.genre}</span>
          <span className="bg-stone-800 text-stone-400 px-2 py-1 rounded">{result.era}</span>
        </div>
      </div>
      <p className="text-stone-300 text-sm leading-relaxed">
        {expanded ? result.raw_text : `${preview}...`}
      </p>
      <button
        onClick={() => setExpanded(!expanded)}
        className="text-amber-500 text-xs mt-2 hover:text-amber-400"
      >
        {expanded ? "Show less" : "Read more"}
      </button>
    </div>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [genre, setGenre] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [error, setError] = useState("");

  async function handleSearch() {
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    setResponse(null);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: 6,
          genre: genre || undefined,
        }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResponse(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-stone-950 text-white">
      {/* Header */}
      <div className="border-b border-stone-800 px-6 py-4">
        <h1 className="text-2xl font-serif text-amber-400">📚 Librarian</h1>
        <p className="text-stone-500 text-sm">Semantic search over classic literature</p>
      </div>

      {/* Search */}
      <div className="max-w-4xl mx-auto px-6 py-10">
        <div className="mb-6">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSearch()}
            placeholder="Search by theme, feeling, or idea... e.g. 'feeling trapped by society's expectations'"
            className="w-full bg-stone-900 border border-stone-700 rounded-lg p-4 text-white placeholder-stone-500 focus:outline-none focus:border-amber-600 resize-none h-24"
          />

          <div className="flex gap-3 mt-3">
            <select
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
              className="bg-stone-900 border border-stone-700 rounded-lg px-3 py-2 text-stone-300 text-sm focus:outline-none focus:border-amber-600"
            >
              <option value="">All genres</option>
              <option value="fiction">Fiction</option>
              <option value="gothic">Gothic</option>
              <option value="fantasy">Fantasy</option>
              <option value="satire">Satire</option>
            </select>

            <button
              onClick={handleSearch}
              disabled={loading || !query.trim()}
              className="bg-amber-700 hover:bg-amber-600 disabled:bg-stone-700 disabled:text-stone-500 text-white px-6 py-2 rounded-lg text-sm font-medium transition-colors"
            >
              {loading ? "Searching..." : "Search"}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900 border border-red-700 text-red-300 rounded-lg p-4 mb-6 text-sm">
            {error}
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="text-center py-12 text-stone-500">
            <div className="text-4xl mb-3">📖</div>
            <p>Searching the stacks...</p>
          </div>
        )}

        {/* Results */}
        {response && (
          <div className="space-y-6">
            {/* Answer */}
            <div className="bg-stone-900 border border-stone-700 rounded-lg p-6">
              <div className="flex justify-between items-center mb-3">
                <h2 className="text-amber-400 font-semibold">Answer</h2>
                <span className="text-stone-500 text-xs">{response.latency_ms.toFixed(0)}ms</span>
              </div>
              <p className="text-stone-200 leading-relaxed mb-4">{response.answer}</p>
              <FaithfulnessBadge faithfulness={response.faithfulness} />
            </div>

            {/* Sources */}
            <div>
              <h2 className="text-stone-400 text-sm font-medium mb-3">
                Retrieved passages — {response.results.length} results
              </h2>
              <div className="space-y-3">
                {response.results.map((r, i) => (
                  <ResultCard key={i} result={r} index={i} />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}