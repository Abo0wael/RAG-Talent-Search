"""
RAG Talent Search Engine - Vector Store

Manages the ChromaDB persistent vector database.
Each resume is stored as ONE document (no arbitrary chunking).
Top-K results map directly to candidates.
"""

import os
import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.config import VECTOR_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from src.embeddings import get_embedding_model
from src.utils import safe_json_value


def _get_chroma_client() -> chromadb.ClientAPI:
    """Get a persistent ChromaDB client."""
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    return chromadb.PersistentClient(
        path=VECTOR_DB_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


def index_exists() -> bool:
    """Check if a ChromaDB index already exists with documents."""
    try:
        client = _get_chroma_client()
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


def build_index(profiles: list[dict], rebuild: bool = False) -> int:
    """
    Build the ChromaDB vector index from parsed resume profiles.

    Each resume is stored as ONE document with its full text and metadata.
    No arbitrary chunking is performed.

    Args:
        profiles: List of normalized CandidateProfile dicts from resume_parser.
        rebuild: If True, delete and recreate the collection.

    Returns:
        Number of documents indexed.
    """
    client = _get_chroma_client()

    if rebuild:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"[VectorStore] Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    # Create or get the collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not rebuild:
        print(f"[VectorStore] Collection already has {collection.count()} documents. "
              f"Use --rebuild to recreate.")
        return collection.count()

    # Prepare documents, metadata, and IDs
    documents = []
    metadatas = []
    ids = []

    for profile in profiles:
        doc_text = profile["text"]
        if not doc_text.strip():
            continue

        doc_id = profile["candidate_id"]

        metadata = {
            "candidate_id": doc_id,
            "name": safe_json_value(profile.get("name")),
            "skills": safe_json_value(profile.get("skills")),
            "designation": safe_json_value(profile.get("designation")),
            "experience": safe_json_value(profile.get("experience")),
            "degree": safe_json_value(profile.get("degree")),
            "college": safe_json_value(profile.get("college")),
            "graduation_year": safe_json_value(profile.get("graduation_year")),
            "companies": safe_json_value(profile.get("companies")),
            "location": safe_json_value(profile.get("location")),
            "email": safe_json_value(profile.get("email")),
        }

        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(doc_id)

    if not documents:
        print("[VectorStore] No documents to index!")
        return 0

    # Generate embeddings
    print(f"[VectorStore] Generating embeddings for {len(documents)} resumes...")
    embedding_model = get_embedding_model()
    embeddings = embedding_model.embed_documents(documents)

    # Add to ChromaDB in batches (ChromaDB has a batch limit)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            ids=ids[i:end],
        )
        print(f"[VectorStore] Indexed batch {i // batch_size + 1} "
              f"({end}/{len(documents)} documents)")

    total = collection.count()
    print(f"[VectorStore] [OK] Index built successfully with {total} documents")
    return total


def load_collection() -> chromadb.Collection:
    """
    Load the existing ChromaDB collection.

    Raises:
        RuntimeError: If the collection doesn't exist or is empty.
    """
    client = _get_chroma_client()
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Vector database not initialized. "
            f"Run: python scripts/build_index.py\n"
            f"Error: {e}"
        )

    if collection.count() == 0:
        raise RuntimeError(
            "Vector database is empty. "
            "Run: python scripts/build_index.py --rebuild"
        )

    return collection


def get_collection_stats() -> dict:
    """Get statistics about the ChromaDB collection."""
    try:
        client = _get_chroma_client()
        collection = client.get_collection(name=COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "document_count": collection.count(),
            "vector_db_path": VECTOR_DB_PATH,
            "embedding_model": EMBEDDING_MODEL,
        }
    except Exception:
        return {
            "collection_name": COLLECTION_NAME,
            "document_count": 0,
            "vector_db_path": VECTOR_DB_PATH,
            "embedding_model": EMBEDDING_MODEL,
            "status": "Not initialized",
        }


def get_all_candidates() -> list[dict]:
    """Retrieve all candidate documents and metadata from the collection."""
    collection = load_collection()
    results = collection.get(include=["documents", "metadatas"])

    candidates = []
    for i, doc_id in enumerate(results["ids"]):
        candidates.append({
            "candidate_id": doc_id,
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
        })

    return candidates


def get_candidate_by_id(candidate_id: str) -> dict | None:
    """Retrieve a specific candidate by ID."""
    collection = load_collection()
    results = collection.get(
        ids=[candidate_id],
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        return None

    return {
        "candidate_id": results["ids"][0],
        "text": results["documents"][0],
        "metadata": results["metadatas"][0],
    }
