# CLAUDE.md

Guidance for Claude Code (and any AI agent) working in this repository.

---

## 1. What this project is

**MediAssist** — a medical question-answering chatbot built on **RAG**
(Retrieval-Augmented Generation).

A medical reference PDF (`data/Medical_book.pdf`, ~16 MB) is chunked, embedded, and
stored in a **Pinecone** vector index. At query time a Flask app retrieves the top-k
relevant chunks and passes them as context to a **Groq-hosted LLaMA 3.3 70B** model,
which produces a short grounded answer rendered in a jQuery chat UI.

The repo is intentionally small and script-shaped (not a package/service framework).
It is a learning/base template intended to be extended.

---

## 2. Repository layout

```
Medical-Chatbot-with-Langchain/
├── app.py                # Flask server + RAG chain (READ PATH)
├── store_index.py        # One-off ingestion script (WRITE PATH)
├── setup.py              # Package metadata (name: medical_chatbot)
├── requirements.txt      # Pinned deps
├── .env                  # Secrets — NOT committed
├── .env.example          # Template for .env
├── data/
│   └── Medical_book.pdf  # Source corpus
├── src/
│   ├── __init__.py       # Empty — makes `src` importable
│   ├── helper.py         # PDF load, metadata filter, chunking, embeddings
│   └── prompt.py         # `system_prompt` string
├── templates/
│   └── chat.html         # Chat UI (Bootstrap + jQuery AJAX)
└── static/
    └── style.css
```

---

## 3. Architecture

### 3.1 Ingestion pipeline — `store_index.py` (run once, or when data changes)

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
   │  download_hugging_face_embeddings()
   │                             all-MiniLM-L6-v2 → 384 dims
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
   │             .as_retriever(search_type="similarity", k=3)
   │             → 3 chunks injected as {context}
   │
   └─ question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
         prompt   = ChatPromptTemplate [ system_prompt(+{context}), human {input} ]
         chatModel = ChatGroq("llama-3.3-70b-versatile", temperature=0.4)
   │
   ▼
response["answer"]  → returned as plain string → rendered in chat bubble
```

**Key invariant:** the embedding model's output dimension (384) must equal the
Pinecone index dimension. Changing the embedding model requires recreating the index.

---

## 4. Key files — what to touch for what

| Want to change | Edit |
| --- | --- |
| Assistant persona / answer length / refusal behavior | `src/prompt.py` (`system_prompt`) |
| Chunk size / overlap | `src/helper.py` → `text_split()` |
| Embedding model (⚠ also change index `dimension`) | `src/helper.py` → `download_hugging_face_embeddings()` and `store_index.py` |
| Number of retrieved chunks (k) / search type | `app.py` → `docsearch.as_retriever(...)` |
| LLM provider, model, temperature | `app.py` → `chatModel` |
| Index name / cloud / region | `app.py` **and** `store_index.py` (`index_name` appears in both — keep in sync) |
| Chat UI, quick-reply chips, styling | `templates/chat.html`, `static/style.css` |
| New API endpoints | `app.py` |
| PDF metadata retained per chunk | `src/helper.py` → `filter_to_minimal_docs()` |

---

## 5. Environment & setup

Python **3.12**. Required env vars (see `.env.example`):

- `PINECONE_API_KEY` — required by both scripts
- `GROQ_API_KEY` — required (active LLM)
- `OPENAI_API_KEY` — optional at runtime, but `store_index.py` currently
  assigns it unguarded, so set it (dummy value is fine) or patch that line

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # then fill in keys

python store_index.py          # one-time: build the Pinecone index
python app.py                  # serves http://localhost:8080
```

`store_index.py` is **not idempotent** — re-running it upserts the same chunks
again with new IDs, creating duplicates. Delete/recreate the index before re-ingesting.

---

## 6. Conventions & style

- Plain scripts, no classes, no framework abstractions. Keep helpers as free
  functions in `src/helper.py`.
- `from src.prompt import *` is used in `app.py` — anything added to `prompt.py`
  becomes a global in `app.py`.
- Imports at top; `load_dotenv()` early; secrets read via `os.environ.get`.
- LangChain style: LCEL-adjacent `create_*_chain` helpers, not manual chain classes.
- Alternative providers are kept as **commented blocks** with banner comments
  (see the OpenAI block in `app.py`). Follow that pattern when adding options.
- No test suite, no linter config, no CI. Verify changes by running the app.

---

## 7. Known issues / gotchas (do not "fix" silently — flag them)

1. **`app.py` duplicates the `PINECONE_API_KEY` read + validation block twice** —
   pure copy-paste; safe to collapse.
2. **`store_index.py` will crash with `TypeError` if `OPENAI_API_KEY` is unset**
   (`os.environ[...] = None`). Guard it like `app.py` does.
3. **`HuggingFaceEmbeddings` is imported from `langchain.embeddings`** (deprecated).
   The modern import is `langchain_huggingface.HuggingFaceEmbeddings`
   (requires adding `langchain-huggingface` to requirements).
4. **`index_name = "medical-chatbot"` is hard-coded in two files.** Changing one
   without the other silently breaks retrieval.
5. **`debug=True` and `host="0.0.0.0"`** — never ship this to production as-is.
6. **`/get` accepts GET and POST but reads `request.form["msg"]`** — a GET without
   a body raises `400`. UI only sends POST.
7. **No conversation memory.** Each request is independent; there is no chat history
   in the chain. Adding memory means switching to
   `create_history_aware_retriever` + a message store.
8. **Answers are returned as a raw string**, not JSON — the frontend inserts it into
   HTML. Escape or sanitize if you ever render model output as markup.
9. **No medical disclaimer** in the prompt or UI. This is a demo, not clinical advice;
   consider adding one before any real-world use.
10. **`data/Medical_book.pdf` is 16 MB and committed to git** — a fresh clone is heavy.

---

## 8. Likely extension directions

If asked to extend this project, these are the natural next steps and where they land:

- **Chat history / multi-turn** → `app.py`: `create_history_aware_retriever`,
  session-keyed message store.
- **Source citations** → return `response["context"]` document `source`/page
  alongside the answer; render in `chat.html`.
- **Streaming responses** → `rag_chain.stream(...)` + Flask SSE endpoint;
  replace `$.ajax` with `EventSource`.
- **Config centralization** → new `src/config.py` reading the optional vars listed
  in `.env.example`; remove hard-coded constants.
- **Multiple / user-uploaded PDFs** → add an upload route + call the
  `store_index.py` pipeline as a reusable function.
- **Deployment** → add `Dockerfile`, swap `app.run()` for gunicorn, `debug=False`.
- **Evaluation** → RAGAS or a simple golden-question set; there is currently nothing.

---

## 9. Working agreements for the agent

- Read `src/helper.py` and `app.py` before changing anything in the RAG path —
  the two files share implicit contracts (embedding dims, index name).
- When changing embeddings or chunking, say explicitly that the Pinecone index
  must be rebuilt.
- Do not commit `.env`, and do not print API key values in output or logs.
- Keep new dependencies pinned in `requirements.txt`, matching the existing style.
- Prefer small, readable edits over refactors — this is a learning codebase.
