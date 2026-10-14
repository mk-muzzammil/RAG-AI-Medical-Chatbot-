"""Select an embedding provider; ingestion and queries must use the same model."""
import os
from pathlib import Path

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384


def local_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=MODEL,
        cache_folder=str(Path(__file__).resolve().parents[1] / ".cache" / "models"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def query_embeddings():
    provider = os.environ.get("EMBEDDING_PROVIDER", "huggingface").lower()
    if provider == "pinecone":
        from src.embeddings_pinecone import PineconeEmbeddings
        return PineconeEmbeddings()
    if provider == "local":
        if os.environ.get("VERCEL"):
            raise ValueError("Local torch embeddings are not supported in this Vercel deployment. Use hosted embeddings.")
        return local_embeddings()
    if provider == "huggingface":
        from src.embeddings_api import HFInferenceEmbeddings
        if os.environ.get("EMBEDDING_MODEL", MODEL) != MODEL:
            raise ValueError("Query embedding model must match the indexed MiniLM model.")
        return HFInferenceEmbeddings(model=MODEL)
    raise ValueError("EMBEDDING_PROVIDER must be pinecone, local, or huggingface.")
