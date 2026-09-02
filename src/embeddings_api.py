"""
API-backed embeddings.

The local `download_hugging_face_embeddings()` helper loads sentence-transformers
and torch (~2 GB installed), which cannot fit inside a Vercel function (250 MB
unzipped limit). This module calls the same model over HTTP instead, so the
runtime dependency footprint is just `requests`.

Same model, same 384 dimensions - the existing Pinecone index stays valid.

Used by app.py (read path). store_index.py (write path) keeps using the local
model, because ingestion runs on your machine where size does not matter.
"""

import os
from typing import List

import requests
from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Current router endpoint. Overridable via HF_EMBED_URL if HuggingFace moves it.
ROUTER_URL = (
    "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
)
# Legacy endpoint, tried automatically if the router returns 404.
LEGACY_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"


def _mean_pool(vec):
    """
    feature-extraction returns either a pooled sentence vector (list[float])
    or token-level vectors (list[list[float]]). Normalise both to list[float].
    """
    if vec and isinstance(vec[0], list):
        n = len(vec)
        return [sum(col) / n for col in zip(*vec)]
    return vec


class HFInferenceEmbeddings(Embeddings):
    """Minimal LangChain Embeddings implementation over the HuggingFace API."""

    def __init__(self, api_key: str = None, model: str = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not self.api_key:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN is not set. Create a free read token at "
                "https://huggingface.co/settings/tokens and add it to your environment."
            )

        self.model = model or os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

        override = os.environ.get("HF_EMBED_URL")
        self.urls = [override] if override else [
            ROUTER_URL.format(model=self.model),
            LEGACY_URL.format(model=self.model),
        ]

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _post(self, inputs: List[str]) -> List[List[float]]:
        payload = {"inputs": inputs, "options": {"wait_for_model": True}}
        last_error = None

        for url in self.urls:
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                continue

            if response.status_code == 404:
                last_error = RuntimeError(f"404 from {url}")
                continue

            response.raise_for_status()
            data = response.json()
            return [_mean_pool(item) for item in data]

        raise RuntimeError(
            f"HuggingFace embedding request failed for model '{self.model}'. "
            f"Last error: {last_error}. Set HF_EMBED_URL to override the endpoint."
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._post(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._post([text])[0]
