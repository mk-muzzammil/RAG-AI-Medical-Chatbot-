"""Pinecone-hosted embeddings; no local ML runtime or HF token required."""
import math
import os

import requests
from langchain_core.embeddings import Embeddings

MODEL = "llama-text-embed-v2"
DIMENSION = 384


class PineconeEmbeddingError(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class PineconeEmbeddings(Embeddings):
    def __init__(self, api_key=None, timeout=20):
        key = api_key or os.environ.get("PINECONE_API_KEY")
        if not key:
            raise ValueError("PINECONE_API_KEY is required.")
        self.timeout = timeout
        self.total_tokens = 0
        self.session = requests.Session()
        self.session.headers.update({"Api-Key": key, "X-Pinecone-API-Version": "2025-04"})

    def _embed(self, texts, input_type):
        vectors = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            try:
                response = self.session.post(
                    "https://api.pinecone.io/embed",
                    json={"model": MODEL, "inputs": [{"text": text} for text in batch],
                          "parameters": {"input_type": input_type, "dimension": DIMENSION,
                                         "truncate": "END"}},
                    timeout=self.timeout,
                )
            except requests.RequestException:
                raise PineconeEmbeddingError("Pinecone embeddings could not be reached. Please try again later.") from None
            if not response.ok:
                messages = {
                    401: "Pinecone authentication failed. Check the configured API key.",
                    403: "Pinecone access was denied. Check API key permissions and project access.",
                    402: "Pinecone embeddings are unavailable due to account limits. Check the account's free-plan allowance.",
                    429: "Pinecone's rate or usage limit has been reached. Try later or wait for the free allowance to reset.",
                }
                raise PineconeEmbeddingError(messages.get(response.status_code,
                    "Pinecone embeddings are temporarily unavailable. Please try again later."), response.status_code)
            try:
                data = response.json()
                values = [item["values"] for item in data["data"]]
                if len(values) != len(batch) or any(
                    len(vector) != DIMENSION or any(
                        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                        for x in vector) for vector in values
                ):
                    raise ValueError("Invalid embeddings")
                self.total_tokens += int(data.get("usage", {}).get("total_tokens", 0))
                vectors.extend(values)
            except (ValueError, TypeError, KeyError):
                raise PineconeEmbeddingError("Pinecone returned invalid embedding vectors.") from None
        return vectors

    def embed_documents(self, texts):
        return self._embed(list(texts), "passage")

    def embed_query(self, text):
        return self._embed([text], "query")[0]
