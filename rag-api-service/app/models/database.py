"""
Database setup and SQLAlchemy ORM models.
"""
import logging
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import Column, Integer, Text, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Declarative Base
Base = declarative_base()


class ChatHistory(Base):
    """
    SQLAlchemy ORM model storing historical chat interactions (user queries and AI responses).
    """
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_query = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    def __repr__(self) -> str:
        return f"<ChatHistory(id={self.id}, timestamp={self.timestamp})>"


# SQLAlchemy Engine and SessionFactory
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a SQLAlchemy database session per request
    and guarantees proper closure upon exit.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initializes database tables defined under the SQLAlchemy Base.
    Logs warnings gracefully if the database host is temporarily unreachable.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables verified/created successfully.")
    except SQLAlchemyError as exc:
        logger.warning(
            "Could not connect to database during table initialization: %s. "
            "Ensure PostgreSQL is running and DATABASE_URL is correct.",
            exc
        )

