# MediAssist

A medical question-answering chatbot built with Flask, LangChain, and RAG (Retrieval-Augmented Generation). It uses a Pinecone vector store populated from medical PDF documents and a Groq-hosted LLaMA model to answer user queries grounded in the retrieved context.

## Technologies Used

- **Python 3** - core language
- **Flask** - web framework serving the chat UI and API
- **LangChain** - orchestration of the RAG pipeline (retrieval, prompt, chains)
- **Pinecone** - vector database for storing and retrieving embeddings
- **sentence-transformers** - HuggingFace embeddings (`all-MiniLM-L6-v2`, 384 dims)

## Prerequisites

- Python 3.12
- A Pinecone account and API key
- A Groq API key
- (Optional) An OpenAI API key if using the OpenAI model path
- Medical PDF files placed in the `data/` directory

## Installation

1. Clone the repository:

```bash
git clone https://github.com/muzammilhussain45/Medical-Chatbot-with-Langchain.git
cd MediAssist
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Create `.env`

Create a `.env` file in the project root with the following variables, replacing the placeholder values with your own keys:

```env
PINECONE_API_KEY="your_pinecone_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
```

## How to Run

### 1. Build the Vector Index (first time only)

Place your medical PDF files in the `data/` folder, then run:

```bash
python store_index.py
```

This loads the PDFs, splits them into text chunks, generates embeddings, and upserts them into the Pinecone index named `medical-chatbot`.

### 2. Start the Application

```bash
python app.py
```

The Flask server starts on `http://0.0.0.0:8080`.

### 3. Use the Chatbot

Open your browser and navigate to:

```
http://localhost:8080
```

Type your medical question in the chat interface and submit. The app retrieves relevant context from Pinecone and generates a concise answer using the Groq-hosted LLaMA model.

## Project Structure

```
MediAssist/
├── app.py              # Flask app and RAG chain setup
├── store_index.py      # Script to build/upsert the Pinecone index
├── setup.py            # Package metadata
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (not committed)
├── data/               # Source PDF documents
├── src/
│   ├── helper.py       # PDF loading, text splitting, embeddings
│   └── prompt.py       # System prompt template
├── templates/
│   └── chat.html       # Chat UI
└── static/             # Static assets
```

## Notes

- The Pinecone index uses 384 dimensions (matching `all-MiniLM-L6-v2`) and the `cosine` metric.
- The active LLM is Groq's `llama-3.3-70b-versatile`. To switch to OpenAI's `Models`, uncomment the `ChatOpenAI` line and comment out the `ChatGroq` line in `app.py`.