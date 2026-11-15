"""
Configuration settings module.
Loads environment variables from .env files and provides central application settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Search candidate locations for .env files
ENV_LOCATIONS = [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
    Path.cwd() / ".env"
]

# Load from the first matching .env file
for env_path in ENV_LOCATIONS:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        break
else:
    # Fallback to standard dotenv search
    load_dotenv()


class Settings:
    """
    Application Settings container holding database, Hugging Face, and model configs.
    """
    PROJECT_NAME: str = "Production RAG API Service"
    VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/rag_db"
    )

    # Hugging Face Configuration
    HUGGINGFACEHUB_API_TOKEN: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

    # LLM & Vector Store Settings
    LLM_REPO_ID: str = os.getenv(
        "LLM_REPO_ID",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    FAISS_INDEX_DIR: Path = BASE_DIR / os.getenv("FAISS_INDEX_PATH", "faiss_index")

    def __init__(self):
        # Sync token into os.environ for third-party libraries (LangChain/Hugging Face)
        if self.HUGGINGFACEHUB_API_TOKEN:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.HUGGINGFACEHUB_API_TOKEN
            os.environ["HF_TOKEN"] = self.HUGGINGFACEHUB_API_TOKEN


settings = Settings()

