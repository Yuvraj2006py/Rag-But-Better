import React, { useState } from "react";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState([]);
  const [faithfulness, setFaithfulness] = useState(null);
  const [hallucinated, setHallucinated] = useState(false);
  const [status, setStatus] = useState("");

  async function handleUpload(e) {
    e.preventDefault();
    if (!files.length) return;
    const form = new FormData();
    Array.from(files).forEach((file) => form.append("files", file));
    setStatus("Uploading...");
    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    setStatus(`Uploaded ${data.uploaded} chunk(s).`);
  }

  async function handleSearch(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setStatus("Searching...");
    const res = await fetch(`${API_BASE}/search/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    setAnswer(data.answer);
    setCitations(data.citations || []);
    setFaithfulness(data.faithfulness);
    setHallucinated(Boolean(data.hallucinated));
    setStatus("Ready");
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>RAG but Better</h1>
        <p>Semantic search + QA over internal docs.</p>
      </header>

      <section className="card">
        <h2>Upload Documents</h2>
        <form onSubmit={handleUpload}>
          <input type="file" multiple onChange={(e) => setFiles(e.target.files)} />
          <button type="submit">Upload</button>
        </form>
      </section>

      <section className="card">
        <h2>Ask a Question</h2>
        <form onSubmit={handleSearch}>
          <input
            type="text"
            placeholder="Ask something..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit">Search</button>
        </form>
        {answer && (
          <div className="answer">
            <h3>Answer</h3>
            <p>{answer}</p>
            <div className="score">
              Faithfulness: {faithfulness ?? "n/a"} | Hallucinated: {hallucinated ? "Yes" : "No"}
            </div>
            {citations.length > 0 && (
              <ul>
                {citations.map((c, i) => (
                  <li key={`${c.chunk_id}-${i}`}>
                    {c.doc_id}#{c.chunk_id}: {c.snippet}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </section>

      <footer className="status">Status: {status || "Idle"}</footer>
    </div>
  );
}
