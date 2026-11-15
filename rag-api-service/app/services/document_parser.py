"""
Document parser service.
Extracts text from PDF byte streams using PyPDF2 and splits content into chunks.
"""
import io
import logging
from typing import List, Union

from PyPDF2 import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def extract_and_split_pdf(
    file_stream: Union[bytes, io.BytesIO],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    source_name: str = "uploaded_document.pdf"
) -> List[Document]:
    """
    Accepts a PDF byte stream or raw bytes, extracts text using PyPDF2,
    and splits the content into manageable chunks using RecursiveCharacterTextSplitter.

    Args:
        file_stream: Raw bytes or BytesIO stream of the PDF file.
        chunk_size: Target token/character chunk size for splitting.
        chunk_overlap: Number of characters to overlap between adjacent chunks.
        source_name: Identifier or filename for document metadata tracking.

    Returns:
        List[Document]: List of LangChain Document objects with page content and metadata.

    Raises:
        ValueError: If the file is empty, corrupted, or contains no readable text.
    """
    if isinstance(file_stream, bytes):
        if not file_stream or len(file_stream) == 0:
            raise ValueError("The provided PDF file byte stream is empty.")
        stream = io.BytesIO(file_stream)
    elif isinstance(file_stream, io.BytesIO):
        stream = file_stream
        stream.seek(0)
    else:
        raise ValueError("Invalid stream type. Expected bytes or io.BytesIO.")

    try:
        reader = PdfReader(stream)
    except Exception as exc:
        logger.error("Failed to parse PDF stream: %s", exc)
        raise ValueError(f"Failed to read PDF file format: {str(exc)}") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("The uploaded PDF contains 0 pages.")

    raw_documents: List[Document] = []
    extracted_any_text = False

    for page_idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                extracted_any_text = True
                raw_documents.append(
                    Document(
                        page_content=page_text.strip(),
                        metadata={
                            "source": source_name,
                            "page": page_idx,
                            "total_pages": total_pages
                        }
                    )
                )
        except Exception as page_exc:
            logger.warning("Error extracting text from page %d: %s", page_idx, page_exc)

    if not extracted_any_text or not raw_documents:
        raise ValueError(
            "Could not extract any readable text from the PDF. "
            "The document might be scanned/image-only or password-protected."
        )

    # Split using LangChain RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(raw_documents)
    logger.info(
        "Successfully extracted %d pages and created %d chunks from '%s'",
        total_pages,
        len(chunks),
        source_name
    )

    return chunks

