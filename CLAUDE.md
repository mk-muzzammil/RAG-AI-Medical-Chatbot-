# CLAUDE.md

Guidance for Claude Code (and any AI agent) working in this repository.

---

## 1. What this project is

**MedCite** — a medical question-answering chatbot built on **RAG**
(Retrieval-Augmented Generation). Formerly named MediAssist; the rename is
complete across the UI, README and package metadata.

A medical reference PDF (`data/Medical_book.pdf`, ~16 MB) is chunked, embedded, and
stored in a **Pinecone** vector index. At query time a Flask app retrieves the top-k
relevant chunks and passes them as context to a **Groq-hosted LLaMA** model, which
produces a short grounded answer rendered in a jQuery chat UI.

The repo is intentionally small and script-shaped (not a package/service framework).
It is a learning/base template intended to be extended, and it is configured to
deploy on Vercel.

---

## 2. Repository layout

```
Medical-Chatbot-with-Langchain/
├── app.py                  # Flask app + RAG chain (READ PATH) — Vercel entrypoint
├── run_local.py            # Dev server — the ONLY place app.run() lives
├── store_index.py          # One-off ingestion script (WRITE PATH) — local only
├── setup.py                # Package metadata (name: medcite)
│
├── requirements.txt        # RUNTIME deps only — what Vercel installs
├── requirements-dev.txt    # INGESTION-only deps (sentence-transformers, pypdf)
│
├── vercel.json             # Function config: maxDuration 60s, excludeFiles
├── .vercelignore           # Keeps data/, venv/, PDFs out of the upload
├── .python-version         # Pins Python 3.12
│
├── .env                    # Secrets — NOT committed
├── .env.example            # Template for .env
│
├── data/
│   └── Medical_book.pdf    # Source corpus (excluded from deployment)
├── src/
│   ├── __init__.py         # Empty — makes `src` importable
│   ├── helper.py           # PDF load, metadata filter, chunking, LOCAL embeddings
│   ├── embeddings_api.py   # HTTP embeddings for the deployed runtime
│   └── prompt.py           # `system_prompt` string
├── templates/
│   └── chat.html           # Chat UI (vanilla CSS + jQuery)
└── static/
    ├── style.css
    └── favicon.svg
```

---

## 3. Architecture

### 3.1 Ingestion pipeline — `store_index.py` (run once, or when data changes)

Runs **only on a developer machine**, never on Vercel.

```
data/*.pdf
   │  load_pdf_file()            DirectoryLoader + PyPDFLoader
   ▼
Document[]  (full metadata)
   │  filter_to_minimal_docs()   keep only metadata={"source": ...}
   ▼
Document[]  (minimal)
   │  text_split()               RecursiveCharacterTextSplitter
   │                             chunk_size=500, chunk_overlap=20
   ▼
chunks
   │  download_hugging_face_embeddings()   [src/helper.py]
   │                             all-MiniLM-L6-v2 LOCALLY → 384 dims
   ▼
Pinecone index "medical-chatbot"
   (created if missing: dim=384, metric=cosine,
    ServerlessSpec aws / us-east-1)
```

### 3.2 Query pipeline — `app.py` (runtime)

```
Browser (templates/chat.html)
   │  jQuery $.ajax POST /get   form field: msg
   ▼
Flask route chat()
   │
   ▼
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
   │
   ├─ retriever: PineconeVectorStore.from_existing_index(...)
   │             .as_retriever(search_type="similarity", k=RETRIEVER_TOP_K)
   │             embeddings: HFInferenceEmbeddings  [src/embeddings_api.py]
   │             → chunks injected as {context}
   │
   └─ question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
         prompt   = ChatPromptTemplate [ system_prompt(+{context}), human {input} ]
         chatModel = ChatGroq(GROQ_MODEL, temperature=LLM_TEMPERATURE)
   │
   ▼
response["answer"]  → returned as plain string → rendered as a text node
```

### 3.3 The two embedding paths — read this before touching either

Both paths use the **same model at the same 384 dimensions**, reached differently:

| Path | Module | Mechanism | Why |
| --- | --- | --- | --- |
| Ingestion | `src/helper.py` → `download_hugging_face_embeddings()` | local `sentence-transformers` | Fast and free across thousands of chunks |
| Query | `src/embeddings_api.py` → `HFInferenceEmbeddings` | HuggingFace Inference API over HTTP | `torch` is ~2 GB; a Vercel function is capped at 250 MB unzipped |

`HFInferenceEmbeddings` is a hand-rolled `langchain_core.embeddings.Embeddings`
subclass using only `requests`. It tries the router endpoint first, falls back to
the legacy endpoint on 404, and honours an `HF_EMBED_URL` override. It mean-pools
when the API returns token-level vectors instead of a pooled sentence vector.

**Key invariant:** both embedding paths and the Pinecone index dimension must all
equal 384. Changing the embedding model means changing both modules *and*
recreating the index.

---

## 4. Key files — what to touch for what

| Want to change | Edit |
| --- | --- |
| Assistant persona / answer length / refusal behavior | `src/prompt.py` (`system_prompt`) |
| Chunk size / overlap | `src/helper.py` → `text_split()` |
| Embedding model (⚠ also change index `dimension`) | `src/helper.py`, `src/embeddings_api.py`, `store_index.py` |
| Number of retrieved chunks (k) | `RETRIEVER_TOP_K` env var (falls back to 3 in `app.py`) |
| LLM model / temperature | `GROQ_MODEL`, `LLM_TEMPERATURE` env vars (fallbacks in `app.py`) |
| Index name | `PINECONE_INDEX_NAME` env var in `app.py`; still hard-coded in `store_index.py` |
| Index cloud / region | `store_index.py` (`ServerlessSpec`) |
| Chat UI, starter questions, styling | `templates/chat.html`, `static/style.css` |
| Brand mark / favicon | inline SVG in `chat.html` header, plus `static/favicon.svg` |
| New API endpoints | `app.py` |
| PDF metadata retained per chunk | `src/helper.py` → `filter_to_minimal_docs()` |
| Deployment size / excluded files | `vercel.json`, `.vercelignore`, `requirements.txt` |

---

## 5. Environment & setup

Python **3.12**. Env vars (see `.env.example`):

Required at runtime: `PINECONE_API_KEY`, `GROQ_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`.
`app.py` validates all three at import and raises a message naming the missing one.

Optional overrides: `GROQ_MODEL`, `PINECONE_INDEX_NAME`, `RETRIEVER_TOP_K`,
`LLM_TEMPERATURE`, `HF_EMBED_URL`, `FLASK_DEBUG`, `PORT`.

```bash
python -m venv venv
venv\Scripts\activate                              # Windows
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                               # then fill in keys

python store_index.py          # one-time: build the Pinecone index
python run_local.py            # serves http://localhost:8080
```

**Never put `app.run()` in `app.py`.** Vercel executes that file with
`python app.py` during the build to discover the WSGI `app` object, so
`__name__` is `"__main__"` there and any server started would hang the deploy
until it times out. The dev server lives in `run_local.py` and nowhere else.

**Nothing in `app.py` may touch the network at import time**, for the same
reason: the build executes the module, so a module-level Pinecone or Groq call
turns a bad key into a failed *build* rather than a failed request. The RAG
chain is therefore built lazily in `get_chain()` on first request, guarded by a
lock.

`store_index.py` is **not idempotent** — re-running it upserts the same chunks
again with new IDs, creating duplicates. Delete/recreate the index before
re-ingesting.

### Deployment

Vercel auto-detects Flask from `app.py` exporting a top-level `app`. No build
command. The four env vars must exist in the Vercel project before the first
deploy. `GET /health` returns the active index name and model — check it first
after deploying, since it fails loudly when configuration is missing.

**Never let `torch`, `sentence-transformers` or `pypdf` into `requirements.txt`.**
That is the single thing that breaks the deployment, and the error appears at
build time as a function-size failure.

---

## 6. Conventions & style

- Plain scripts, no classes, no framework abstractions — except
  `HFInferenceEmbeddings`, which must be a class to satisfy the LangChain
  `Embeddings` interface. Keep other helpers as free functions in `src/helper.py`.
- `from src.prompt import *` is used in `app.py` — anything added to `prompt.py`
  becomes a global in `app.py`.
- Imports at top; `load_dotenv()` early; secrets read via `os.environ.get`.
- Config reads follow `os.environ.get("NAME", default)` so behavior is tunable
  without code edits, which matters on Vercel where redeploying is the only way
  to change code but env vars can change freely.
- LangChain style: LCEL-adjacent `create_*_chain` helpers, not manual chain classes.
- Alternative providers are kept as **commented blocks** with banner comments
  (see the OpenAI block in `app.py`). Follow that pattern when adding options.
- Frontend has no build step and no CSS framework. Vanilla CSS with custom
  properties in `:root`, inline SVG icons, jQuery from CDN. Brand color `#DEA535`,
  typeface Roboto.
- Model output is inserted with jQuery `.text()`, never string-interpolated into
  HTML. Keep it that way.
- No test suite, no linter config, no CI. Verify changes by running the app.

---

## 7. Known issues / gotchas (do not "fix" silently — flag them)

1. **`index_name` is hard-coded in `store_index.py`** while `app.py` reads
   `PINECONE_INDEX_NAME`. Changing one without the other silently breaks retrieval.
2. **`HuggingFaceEmbeddings` is imported from `langchain.embeddings`** in
   `src/helper.py` (deprecated). The modern import is
   `langchain_huggingface.HuggingFaceEmbeddings`. Emits two warnings on startup.
   Ingestion-side only — the deployed path does not touch it.
3. **`store_index.py` re-run duplicates chunks.** Not idempotent.
4. **No conversation memory.** Each request is independent. Adding memory means
   switching to `create_history_aware_retriever` + a message store.
5. **No source citations** despite the project name. `response["context"]` holds
   the retrieved documents with their `source` metadata; nothing surfaces them.
   This is the most obvious next feature.
6. **HuggingFace serverless inference is the runtime's weak link.** It rate-limits
   on the free tier and has moved endpoints before. `HF_EMBED_URL` exists so a
   move can be fixed without a code change.
7. **First request after a cold start is slow** (5–15s) — function boot plus HF
   model wake-up. `wait_for_model` is set, so it waits rather than failing.
8. **`data/Medical_book.pdf` is 16 MB and committed to git** — a fresh clone is
   heavy. It is excluded from the deployment bundle, not from the repo.
9. **Groq model IDs get retired.** A `NotFoundError: model does not exist or you do
   not have access` means the ID in `GROQ_MODEL` is dead for that account — list
   models with the account's key and pick a live one.
10. **This is a demo, not clinical advice.** The UI carries a disclaimer in the
    composer hint. Do not remove it.

### Previously flagged, now fixed — do not re-report

- Duplicated `PINECONE_API_KEY` validation block in `app.py` — collapsed into `_require()`.
- `store_index.py` crashing with `TypeError` when `OPENAI_API_KEY` was unset — guarded.
- `debug=True` and `host="0.0.0.0"` — now `127.0.0.1`, debug via `FLASK_DEBUG`.
- `/get` accepting GET and raising 400 — now POST-only with a real error path.
- Model output interpolated into HTML — now inserted as a text node.
- No medical disclaimer — added to the UI.

---

## 8. Likely extension directions

- **Source citations** → return `response["context"]` document `source`/page
  alongside the answer; render under each bot bubble in `chat.html`. Delivers on
  the project's name.
- **Chat history / multi-turn** → `app.py`: `create_history_aware_retriever`,
  session-keyed message store. Note: serverless functions are stateless, so this
  needs an external store (Vercel KV, Redis) rather than process memory.
- **Streaming responses** → `rag_chain.stream(...)` + SSE endpoint; replace
  `$.ajax` with `EventSource`. Vercel Python functions support streaming.
- **Config centralization** → new `src/config.py` collecting the `os.environ.get`
  calls now spread across `app.py` and `src/embeddings_api.py`.
- **Multiple / user-uploaded PDFs** → an upload route plus the ingestion pipeline
  refactored into a reusable function. Ingestion cannot run in the Vercel
  function, so this needs a separate worker or a local step.
- **Evaluation** → RAGAS or a golden-question set; there is currently nothing.

---

## 9. Working agreements for the agent

- Read `src/helper.py`, `src/embeddings_api.py` and `app.py` before changing
  anything in the RAG path — the three share implicit contracts (embedding
  dimension, index name).
- When changing embeddings or chunking, say explicitly that the Pinecone index
  must be rebuilt.
- Before adding any dependency, check whether it is needed at runtime or only for
  ingestion, and put it in the right requirements file. Runtime additions must be
  weighed against the 250 MB function limit.
- Do not commit `.env`, and do not print API key values in output or logs.
- Keep new dependencies pinned, matching the existing style.
- Prefer small, readable edits over refactors — this is a learning codebase.
