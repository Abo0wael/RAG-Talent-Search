"""
RAG Talent Search Engine - Embeddings

Provides a LangChain-compatible embedding wrapper around
sentence-transformers/all-MiniLM-L6-v2 (runs locally, no API key needed).
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL


_embedding_instance: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Get or create the singleton embedding model.

    Uses sentence-transformers/all-MiniLM-L6-v2 by default.
    The model is downloaded on first use and cached locally.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    global _embedding_instance

    if _embedding_instance is None:
        print(f"[Embeddings] Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"[Embeddings] Model loaded successfully")

    return _embedding_instance


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Args:
        query: The search query text.

    Returns:
        List of floats representing the embedding vector.
    """
    model = get_embedding_model()
    return model.embed_query(query)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of document texts.

    Args:
        texts: List of document text strings.

    Returns:
        List of embedding vectors.
    """
    model = get_embedding_model()
    return model.embed_documents(texts)
