"""
LLM Chain service.
Configures HuggingFaceEndpoint LLM and builds the LangChain RetrievalQA chain
connected to the FAISS vector retriever.
"""
import logging
from typing import Any, Dict, Optional

try:
    from langchain_huggingface import HuggingFaceEndpoint
except ImportError:
    from langchain_community.llms import HuggingFaceEndpoint

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

from app.core.config import settings
from app.services.vector_store import get_retriever

logger = logging.getLogger(__name__)

# Production-grade Prompt Template
RAG_PROMPT_TEMPLATE = """You are an expert AI assistant answering questions based strictly on the provided context.
Use ONLY the following context to answer the question.
If the answer is not contained in the context, state clearly: "I cannot find the answer in the provided document."
Do not make up facts or extrapolate beyond the text.

Context:
{context}

Question:
{question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def get_llm(
    repo_id: Optional[str] = None,
    temperature: float = 0.1,
    max_new_tokens: int = 512
) -> HuggingFaceEndpoint:
    """
    Instantiates a HuggingFaceEndpoint LLM client.

    Args:
        repo_id: Hugging Face model repository ID. Defaults to settings.LLM_REPO_ID.
        temperature: Sampling temperature for text generation.
        max_new_tokens: Maximum tokens to generate in response.

    Returns:
        HuggingFaceEndpoint: Configured LangChain LLM instance.

    Raises:
        ValueError: If HUGGINGFACEHUB_API_TOKEN is missing.
    """
    if not settings.HUGGINGFACEHUB_API_TOKEN:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN is not configured in .env or environment variables."
        )

    model_repo = repo_id or settings.LLM_REPO_ID
    logger.info("Initializing HuggingFaceEndpoint with model '%s'", model_repo)

    return HuggingFaceEndpoint(
        repo_id=model_repo,
        huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.05
    )


def build_retrieval_qa_chain(
    retriever: VectorStoreRetriever,
    repo_id: Optional[str] = None
) -> RetrievalQA:
    """
    Constructs a LangChain RetrievalQA chain bound to the FAISS retriever and Hugging Face LLM.

    Args:
        retriever: The LangChain VectorStoreRetriever instance.
        repo_id: Optional model override.

    Returns:
        RetrievalQA: An executable LangChain QA chain.
    """
    llm = get_llm(repo_id=repo_id)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain


def execute_rag_query(question: str) -> Dict[str, Any]:
    """
    High-level helper to execute a question against the active FAISS retriever and QA chain.

    Args:
        question: User query string.

    Returns:
        Dict[str, Any]: Response containing 'answer', 'sources', and 'raw_result'.

    Raises:
        FileNotFoundError: If the vector index has not been populated.
        Exception: On LLM or retrieval execution errors.
    """
    retriever = get_retriever()
    chain = build_retrieval_qa_chain(retriever=retriever)

    # Invoke chain
    result = chain.invoke({"query": question})

    raw_answer = result.get("result", "")
    source_docs = result.get("source_documents", [])

    sources_metadata = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
            "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        }
        for doc in source_docs
    ]

    return {
        "question": question,
        "answer": raw_answer.strip(),
        "sources": sources_metadata
    }

