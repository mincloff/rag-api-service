"""
Main FastAPI application entry point.
Initializes the FastAPI app, manages database startup lifecycle,
configures CORS middleware, and mounts API routers.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.core.config import settings
from app.models.database import init_db
from app.services.vector_store import is_index_ready

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rag-api-service")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    Handles startup tasks (such as initializing database tables)
    and graceful shutdown.
    """
    logger.info("Starting up %s (version: %s)...", settings.PROJECT_NAME, settings.VERSION)
    # Initialize SQLAlchemy database tables if not already present
    init_db()
    yield
    logger.info("Shutting down %s...", settings.PROJECT_NAME)


# Create FastAPI instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade Retrieval-Augmented Generation (RAG) API using FastAPI, LangChain, FAISS, and PostgreSQL.",
    lifespan=lifespan
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount RAG endpoints router
app.include_router(api_router)


@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    """
    Root welcome endpoint with API documentation links.
    """
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "status": "operational"
    }


@app.get("/health", tags=["Health"], status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """
    Service health check endpoint reporting service and vector index status.
    """
    index_status = "ready" if is_index_ready() else "not_initialized"
    return {
        "status": "healthy",
        "vector_store_status": index_status,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_repo_id": settings.LLM_REPO_ID
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

