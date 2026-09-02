from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)

load_dotenv()


def _require(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"{name} environment variable is not set. "
            "Locally: copy .env.example to .env and fill it in. "
            "On Vercel: add it under Project Settings > Environment Variables."
        )
    return value


PINECONE_API_KEY = _require("PINECONE_API_KEY")
GROQ_API_KEY = _require("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = _require("HUGGINGFACEHUB_API_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medical-chatbot")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.environ.get("RETRIEVER_TOP_K", "3"))


# ==========================================================
# Embeddings
# ----------------------------------------------------------
# The read path uses the HuggingFace Inference API rather than a local
# sentence-transformers model, so the deployed bundle stays small enough for a
# serverless function. Same model, same 384 dimensions as the built index.
# store_index.py still embeds locally during ingestion.
# ==========================================================
from src.embeddings_api import HFInferenceEmbeddings

embeddings = HFInferenceEmbeddings(api_key=HUGGINGFACEHUB_API_TOKEN)

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

# ==========================================
# OpenAI Model Initialization (Commented out)
# ==========================================
# from langchain_openai import ChatOpenAI
# chatModel = ChatOpenAI(model="gpt-4o")

# ==========================================
# Groq Model Initialization (Active)
# ==========================================
chatModel = ChatGroq(
    model=GROQ_MODEL,
    temperature=float(os.environ.get("LLM_TEMPERATURE", "0.4")),
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/health")
def health():
    return {"status": "ok", "index": INDEX_NAME, "model": GROQ_MODEL}


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please enter a question.", 400

    try:
        response = rag_chain.invoke({"input": msg})
    except Exception as exc:
        app.logger.exception("RAG chain failed")
        return f"Upstream error: {type(exc).__name__}", 502

    return str(response["answer"])


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 8080)),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
    )
