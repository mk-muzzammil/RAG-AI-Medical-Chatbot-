"""Index one PDF with local or Pinecone embeddings and repeatable vector IDs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from langchain_core.documents import Document
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec

from src.embeddings import MODEL, DIMENSION, local_embeddings
from src.helper import text_split

ROOT = Path(__file__).resolve().parent


def prepare_book(path, chunk_size=1000, chunk_overlap=150, model=MODEL):
    path = Path(path)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = PdfReader(path)
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(Document(page_content=text, metadata={
                "source": path.name, "page": page_number, "book_sha256": fingerprint,
                "embedding_model": model,
            }))
    if not documents:
        raise ValueError("The PDF has no extractable text. OCR is required before indexing.")
    chunks = text_split(documents, chunk_size, chunk_overlap)
    # Version the namespace so changed books/chunking never mix with old vectors.
    signature = f"{fingerprint}:{model}:text-v1:{chunk_size}:{chunk_overlap}"
    namespace = "medical-book-" + hashlib.sha256(signature.encode()).hexdigest()[:16]
    ids = [f"{namespace}-{i:06d}" for i in range(len(chunks))]
    report = {"source": path.name, "sha256": fingerprint, "pages": len(reader.pages),
              "text_pages": len(documents), "chunks": len(chunks), "namespace": namespace,
              "model": model, "dimension": DIMENSION,
              "chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    return chunks, ids, report


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "data" / "Medical_book.pdf")
    parser.add_argument("--dry-run", action="store_true", help="Extract and count chunks; do not download models or call Pinecone")
    parser.add_argument("--provider", choices=["local", "pinecone"], default="local")
    args = parser.parse_args()
    from src.embeddings_pinecone import MODEL as PINECONE_MODEL, PineconeEmbeddings
    model = PINECONE_MODEL if args.provider == "pinecone" else MODEL
    chunks, ids, report = prepare_book(args.pdf, model=model)
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
    report["index"] = index_name
    print(json.dumps(report, indent=2), flush=True)
    if args.dry_run:
        return
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY is required for indexing.")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(index_name):
        pc.create_index(name=index_name, dimension=DIMENSION, metric="cosine",
                        spec=ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"),
                                            region=os.getenv("PINECONE_REGION", "us-east-1")))
    deadline = time.monotonic() + 120
    while True:
        description = pc.describe_index(index_name)
        if description.dimension != DIMENSION or description.metric != "cosine":
            raise ValueError("Index must use 384 dimensions and cosine similarity. Existing index was not changed.")
        if description.status["ready"]:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Pinecone index did not become ready within 120 seconds.")
        time.sleep(2)
    index = pc.Index(host=description.host)
    embeddings = PineconeEmbeddings(timeout=60) if args.provider == "pinecone" else local_embeddings()
    namespace = report["namespace"]
    for start in range(0, len(chunks), 64):
        batch_ids = ids[start:start + 64]
        # Resume without re-embedding already uploaded chunks or wasting tokens.
        existing = index.fetch(ids=batch_ids, namespace=namespace).vectors
        pending = [(id_, doc) for id_, doc in zip(batch_ids, chunks[start:start + 64]) if id_ not in existing]
        if not pending:
            print(f"Already indexed {min(start + 64, len(chunks))}/{len(chunks)} chunks", flush=True)
            continue
        batch = [doc for _, doc in pending]
        vectors = embeddings.embed_documents([doc.page_content for doc in batch])
        if any(len(vector) != DIMENSION for vector in vectors):
            raise ValueError("Embedding dimension does not match the index.")
        index.upsert(namespace=namespace, vectors=[
            {"id": id_, "values": vector,
             "metadata": {**doc.metadata, "text": doc.page_content}}
            for (id_, doc), vector in zip(pending, vectors)
        ])
        print(f"Indexed {min(start + 64, len(chunks))}/{len(chunks)} chunks", flush=True)
        if args.provider == "pinecone":
            # Pace ingestion; request-path embeddings never sleep or retry.
            time.sleep(6)
    # Pinecone is eventually consistent; wait for the full namespace to be visible.
    deadline = time.monotonic() + 120
    while True:
        stats = index.describe_index_stats()
        count = stats.namespaces.get(namespace, {}).get("vector_count", 0)
        if count == len(chunks):
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"Uploaded chunks, but only {count} are visible; rerun safely to verify.")
        time.sleep(2)
    report["indexed_vectors"] = count
    report["embedding_tokens_this_run"] = getattr(embeddings, "total_tokens", None)
    (ROOT / ".cache").mkdir(exist_ok=True)
    (ROOT / ".cache" / "index-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {count} vectors. Set PINECONE_NAMESPACE={namespace} in the app environment.", flush=True)


if __name__ == "__main__":
    main()
