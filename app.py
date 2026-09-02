from flask import Flask, render_template, request
from dotenv import load_dotenv
from src.prompt import *
import os
import threading


app = Flask(__name__)

load_dotenv()

# ----------------------------------------------------------
# Configuration. Read at import (cheap, no network), so a missing
# variable is visible on /health without paying for a chain build.
# ----------------------------------------------------------
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.environ.get("RETRIEVER_TOP_K", "3"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.4"))

REQUIRED_VARS = ("PINECONE_API_KEY", "GROQ_API_KEY", "HUGGINGFACEHUB_API_TOKEN")


def missing_vars():
    return [name for name in REQUIRED_VARS if not os.environ.get(name)]


# ----------------------------------------------------------
# The RAG chain is built lazily on the first request, never at import.
#
# Vercel executes this module during the build to discover the WSGI app.
# Anything that touches the network at import time turns a bad key or a
# transient outage into a failed BUILD instead of a failed request, and
# repeats on every cold start before the platform can serve anything.
# ----------------------------------------------------------
_chain = None
_chain_lock = threading.Lock()


def build_chain():
    from langchain_pinecone import PineconeVectorStore
    from langchain_groq import ChatGroq
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from src.embeddings_api import HFInferenceEmbeddings

    absent = missing_vars()
    if absent:
        raise RuntimeError(
            "Missing environment variable(s): "
            + ", ".join(absent)
            + ". Locally, copy .env.example to .env. On Vercel, add them under "
            "Project Settings > Environment Variables (raw value, no quotes)."
        )

    embeddings = HFInferenceEmbeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    retriever = docsearch.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )

    # ==========================================
    # OpenAI Model Initialization (Commented out)
    # ==========================================
    # from langchain_openai import ChatOpenAI
    # chatModel = ChatOpenAI(model="gpt-4o")

    # ==========================================
    # Groq Model Initialization (Active)
    # ==========================================
    chatModel = ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def get_chain():
    global _chain
    if _chain is None:
        with _chain_lock:
            if _chain is None:
                _chain = build_chain()
    return _chain


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health")
def health():
    """
    Config check. By default this is cheap - it does not build the chain or
    call any API. Use it first after a deploy.

    /health?deep=1 additionally embeds a short test string, which exercises
    the HuggingFace token and reports exactly what failed. Use it to diagnose
    auth problems without going through the chat UI.
    """
    absent = missing_vars()
    body = {
        "status": "ok" if not absent else "misconfigured",
        "missing_env": absent,
        "index": INDEX_NAME,
        "model": GROQ_MODEL,
        "top_k": TOP_K,
        "chain_built": _chain is not None,
    }

    if request.args.get("deep") and not absent:
        try:
            from src.embeddings_api import HFInferenceEmbeddings

            vector = HFInferenceEmbeddings().embed_query("health check")
            body["embeddings"] = {"ok": True, "dimensions": len(vector)}
            if len(vector) != 384:
                body["status"] = "misconfigured"
                body["embeddings"]["warning"] = (
                    f"Expected 384 dimensions to match the Pinecone index, got {len(vector)}."
                )
        except Exception as exc:
            body["status"] = "misconfigured"
            body["embeddings"] = {"ok": False, "error": str(exc)}

    return body, (200 if body["status"] == "ok" else 503)


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a question.", 400

    try:
        chain = get_chain()
    except Exception as exc:
        app.logger.exception("Chain initialization failed")
        return f"Service not configured: {exc}", 503

    try:
        response = chain.invoke({"input": msg})
    except Exception as exc:
        app.logger.exception("RAG chain failed")
        return f"Upstream error: {type(exc).__name__}: {exc}", 502

    return str(response["answer"])


# ----------------------------------------------------------
# NOTE: there is deliberately no `app.run()` in this file.
#
# Vercel executes this module with `python app.py` to discover the WSGI `app`
# object, which means __name__ IS "__main__" during the build. Any server
# started here would run forever and hang the deploy.
#
# To serve locally, run:  python run_local.py
# ----------------------------------------------------------
