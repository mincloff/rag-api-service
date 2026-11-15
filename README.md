# Production-Grade RAG API Service

A microservice-oriented Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain (LCEL), and FAISS. This service parses PDF documents, indexes text chunks into a local vector store, and utilizes Google's Gemini models for high-accuracy, context-aware question answering. All interactions are logged to a PostgreSQL database for audit and history tracking.

## Architecture

The API operates on a dual-pipeline architecture:

1. **Ingestion Pipeline (`/upload`)**:
   Accepts a PDF, extracts text using PyPDF2, chunks the content using LangChain's `RecursiveCharacterTextSplitter`, and embeds the chunks into a local FAISS vector database using `sentence-transformers/all-MiniLM-L6-v2`.
2. **Retrieval & Generation Pipeline (`/ask`)**:
   Accepts a natural language query, performs a similarity search against the FAISS index to retrieve the top $k$ relevant chunks, and routes the context and query through a modern LangChain Expression Language (LCEL) chain to Google Gemini (3.8 Flash) for generation.

## Tech Stack

- **Framework:** FastAPI, Uvicorn (Asynchronous ASGI)
- **Orchestration:** LangChain (LCEL architecture)
- **LLM Provider:** Google GenAI (Gemini 3.8 Flash)
- **Embeddings & Vector Store:** HuggingFace `all-MiniLM-L6-v2`, FAISS (Local CPU)
- **Database & ORM:** PostgreSQL, SQLAlchemy (psycopg-binary)

## Project Structure

```text
rag-api-service/
├── app/
│   ├── api/
│   │   └── endpoints.py        # FastAPI routers for /upload, /ask, /history
│   ├── core/
│   │   └── config.py           # Environment and pydantic settings
│   ├── models/
│   │   └── database.py         # SQLAlchemy Base and ChatHistory schema
│   ├── services/
│   │   ├── document_parser.py  # PDF extraction and chunking
│   │   ├── vector_store.py     # FAISS embedding and indexing
│   │   └── llm_chain.py        # LCEL retrieval QA pipeline
│   └── main.py                 # Application lifecycle and middleware
├── requirements.txt
├── .env.example
└── .gitignore
```

## Local Development Setup

### 1. Prerequisites

- Python 3.12+
- PostgreSQL server running locally

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone [https://github.com/mincloff/rag-api-service.git](https://github.com/mincloff/rag-api-service.git)
cd rag-api-service
python -m venv venv

# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@localhost:5432/rag_db
GOOGLE_API_KEY=your_gemini_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=faiss_index
```

### 4. Running the Service

Start the FastAPI server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000/docs` to interact with the auto-generated Swagger UI.

## API Reference

- `POST /upload`
  - **Payload:** `multipart/form-data` (PDF file)
  - **Action:** Parses and indexes the document into FAISS.
- `POST /ask`
  - **Payload:** `{"question": "string"}`
  - **Action:** Retrieves context, queries Gemini, logs the transaction to PostgreSQL, and returns the generated answer with source snippets.
- `GET /history`
  - **Action:** Retrieves the chronological log of all QA interactions from PostgreSQL.
