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
import math
from typing import List

import requests
from langchain_core.embeddings import Embeddings

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Current router endpoint. Overridable via HF_EMBED_URL if HuggingFace moves it.
ROUTER_URL = (
    "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
)
class EmbeddingServiceError(RuntimeError):
    """Controlled, public-safe embedding failure."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


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

    def __init__(self, api_key: str = None, model: str = None, timeout: int = 15):
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
        ]

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _post(self, inputs: List[str]) -> List[List[float]]:
        payload = {"inputs": inputs, "options": {"wait_for_model": True}}
        try:
            response = self.session.post(self.urls[0], json=payload, timeout=self.timeout)
        except requests.RequestException:
            raise EmbeddingServiceError(
                "The medical-book embedding service could not be reached. Please try again later."
            ) from None

        errors = {
            401: "Hugging Face authentication failed. The administrator must update the inference token.",
            403: "Hugging Face inference access was denied. The administrator must check token permissions.",
            402: "Medical-book search is unavailable because Hugging Face inference credits are exhausted. The administrator must restore inference credits or configure a compatible embedding endpoint.",
            404: "The embedding model endpoint is unavailable. The administrator must configure HF_EMBED_URL for the same embedding model.",
            410: "The embedding endpoint has been retired. The administrator must configure HF_EMBED_URL for the same embedding model.",
            429: "The medical-book embedding service is rate-limited. Please try again later.",
        }
        if not response.ok:
            raise EmbeddingServiceError(
                errors.get(response.status_code, "The medical-book embedding service is temporarily unavailable. Please try again later."),
                response.status_code,
            )

        try:
            data = response.json()
            # Some endpoints omit the batch dimension for a single input.
            if len(inputs) == 1 and isinstance(data, list) and data and isinstance(data[0], (float, int)):
                data = [data]
            if not isinstance(data, list) or len(data) != len(inputs):
                raise ValueError("Unexpected batch shape")
            vectors = [_mean_pool(item) for item in data]
            if any(not isinstance(v, list) or len(v) != 384 or
                   any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in v)
                   for v in vectors):
                raise ValueError("Expected finite 384-dimensional vectors")
            return vectors
        except (ValueError, TypeError, IndexError):
            raise EmbeddingServiceError(
                "The embedding service returned an invalid vector. The administrator must verify the endpoint serves all-MiniLM-L6-v2 with 384 dimensions."
            ) from None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._post(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._post([text])[0]
