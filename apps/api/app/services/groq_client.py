from __future__ import annotations

import httpx

from app.core.config import settings


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self.base_url = "https://api.groq.com/openai/v1"

    def rewrite_query(self, query: str) -> str:
        if not self.api_key:
            return query
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Rewrite the query for retrieval, keep intent."},
                {"role": "user", "content": query},
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=20) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def answer(self, query: str, context: str) -> str:
        if not self.api_key:
            return "Not enough information."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Answer only using the provided context."},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=20) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
