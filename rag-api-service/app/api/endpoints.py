"""
API routes for document upload and RAG question answering.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.models.database import ChatHistory, get_db
from app.services.document_parser import extract_and_split_pdf
from app.services.vector_store import index_chunks, is_index_ready
from app.services.llm_chain import execute_rag_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["RAG Endpoints"])


# --- Schemas ---

class UploadResponse(BaseModel):
    """Schema for document upload response."""
    status: str = Field(default="success")
    message: str
    filename: str
    chunks_indexed: int


class AskRequest(BaseModel):
    """Schema for question request."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer using the indexed RAG context."
    )


class AskResponse(BaseModel):
    """Schema for question response."""
    question: str
    answer: str
    timestamp: datetime
    sources: Optional[List[Dict[str, Any]]] = None


class ChatHistoryItem(BaseModel):
    """Schema for persisted chat history item."""
    id: int
    user_query: str
    ai_response: str
    timestamp: datetime

    class Config:
        from_attributes = True


# --- Endpoints ---

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and index a PDF document",
    description="Accepts a PDF file upload, extracts and chunks text using PyPDF2, and updates the local FAISS index."
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF document to parse and index")
) -> UploadResponse:
    """
    Handles PDF file uploads:
    1. Validates that the uploaded file is a PDF.
    2. Reads and parses the byte stream into text chunks.
    3. Indexes the chunks into the local FAISS vector store.
    """
    filename = file.filename or "unknown.pdf"

    # Validation: File extension check
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type for '{filename}'. Only PDF files are supported."
        )

    try:
        content = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file stream: {str(exc)}"
        ) from exc

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF file is empty (0 bytes)."
        )

    # Step 1: Parse and chunk PDF
    try:
        chunks = extract_and_split_pdf(file_stream=content, source_name=filename)
    except ValueError as val_err:
        logger.warning("PDF parsing validation error for '%s': %s", filename, val_err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        ) from val_err
    except Exception as exc:
        logger.error("Unexpected error parsing PDF '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while parsing the PDF: {str(exc)}"
        ) from exc

    # Step 2: Index chunks into FAISS
    try:
        index_chunks(chunks)
    except Exception as exc:
        logger.error("Failed to index chunks into FAISS: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store indexing failed: {str(exc)}"
        ) from exc

    return UploadResponse(
        status="success",
        message=f"Document '{filename}' successfully parsed and indexed.",
        filename=filename,
        chunks_indexed=len(chunks)
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question against the indexed documents",
    description="Queries the FAISS retriever, generates an answer via HuggingFace LLM, and records the interaction to PostgreSQL."
)
async def ask_question(
    payload: AskRequest,
    db: Session = Depends(get_db)
) -> AskResponse:
    """
    Handles RAG QA queries:
    1. Checks if a FAISS vector index has been created.
    2. Retrieves relevant chunks and invokes the LLM QA chain.
    3. Persists user query, AI response, and timestamp into the PostgreSQL database.
    4. Returns the AI generated answer with metadata.
    """
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The question field cannot be empty or whitespace."
        )

    # Check vector index readiness
    if not is_index_ready():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Please upload a PDF using the /upload endpoint first."
        )

    # Step 1: Execute RAG Retrieval & LLM Generation
    try:
        rag_output = execute_rag_query(question=clean_question)
        answer = rag_output["answer"]
        sources = rag_output.get("sources", [])
    except FileNotFoundError as fnf_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(fnf_err)
        ) from fnf_err
    except Exception as exc:
        logger.error("RAG pipeline inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing RAG chain: {str(exc)}"
        ) from exc

    current_timestamp = datetime.now(timezone.utc)

    # Step 2: Log query and response into PostgreSQL
    try:
        chat_log = ChatHistory(
            user_query=clean_question,
            ai_response=answer,
            timestamp=current_timestamp
        )
        db.add(chat_log)
        db.commit()
        db.refresh(chat_log)
        logger.info("Saved query/response record (id=%d) to database.", chat_log.id)
    except SQLAlchemyError as db_err:
        db.rollback()
        logger.error("Failed to record chat history to database: %s", db_err)
        # Note: We still return the answer so user interaction doesn't fail,
        # but we log the database persistence error.

    return AskResponse(
        question=clean_question,
        answer=answer,
        timestamp=current_timestamp,
        sources=sources
    )


@router.get(
    "/history",
    response_model=List[ChatHistoryItem],
    status_code=status.HTTP_200_OK,
    summary="Get recent chat history",
    description="Retrieves the most recent queries and responses logged in PostgreSQL."
)
def get_chat_history(
    limit: int = 50,
    db: Session = Depends(get_db)
) -> List[ChatHistoryItem]:
    """
    Fetches historical interactions from PostgreSQL ordered from newest to oldest.
    """
    try:
        records = (
            db.query(ChatHistory)
            .order_by(ChatHistory.id.desc())
            .limit(limit)
            .all()
        )
        return records
    except SQLAlchemyError as exc:
        logger.error("Error reading chat history: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history from database."
        ) from exc

