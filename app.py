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
    Cheap config check - does not build the chain or call any API.
    Use this first after a deploy.
    """
    absent = missing_vars()
    return {
        "status": "ok" if not absent else "misconfigured",
        "missing_env": absent,
        "index": INDEX_NAME,
        "model": GROQ_MODEL,
        "top_k": TOP_K,
        "chain_built": _chain is not None,
    }, (200 if not absent else 503)


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
# Local development server only.
#
# Vercel executes this file with `python app.py` to discover the WSGI `app`
# object, so __name__ IS "__main__" there. Without the VERCEL guard the dev
# server starts during the build and never exits, hanging the deploy.
# Vercel sets VERCEL=1 in both build and runtime environments.
# ----------------------------------------------------------
if __name__ == "__main__" and not os.environ.get("VERCEL"):
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 8080)),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
    )
