"""
RAG Talent Search Engine - Semantic Retriever

Retrieves candidate resumes from ChromaDB based on semantic similarity.
Labels scores explicitly as "Semantic Similarity Score".
"""

import time
from typing import Any

from src.config import TOP_K
from src.embeddings import embed_query
from src.vector_store import load_collection
from src.utils import timer


def _distance_to_similarity(distance: float) -> float:
    """
    Convert ChromaDB cosine distance to a normalized similarity score.
    ChromaDB cosine distance is in [0, 2], where 0 is identical and 2 is opposite.
    We convert this to a similarity score roughly in [0, 1]:
      similarity = max(0.0, 1.0 - (distance / 2.0))
    or standard cosine similarity: 1.0 - distance.
    Here we use max(0.0, 1.0 - distance) for standard normalized cosine similarity.
    """
    # For normalized vectors, cosine distance is 1 - cosine_similarity.
    # Therefore, cosine_similarity = 1 - distance.
    sim = 1.0 - distance
    return max(0.0, min(1.0, float(sim)))


def retrieve_candidates(
    query: str,
    top_k: int = TOP_K,
    collection=None
) -> tuple[list[dict[str, Any]], float]:
    """
    Retrieve Top-K candidate resumes for a recruiter query.

    Args:
        query: Natural language search query.
        top_k: Number of candidates to retrieve.
        collection: Optional pre-loaded ChromaDB collection.

    Returns:
        tuple of (results_list, latency_in_seconds)
        Each item in results_list:
        {
            "candidate_id": str,
            "semantic_similarity_score": float,
            "document": str,
            "metadata": dict
        }
    """
    if not query or not query.strip():
        return [], 0.0

    start_time = time.perf_counter()

    if collection is None:
        collection = load_collection()

    # Generate query embedding
    query_vector = embed_query(query.strip())

    # Query ChromaDB
    raw_results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    elapsed = time.perf_counter() - start_time

    results = []
    if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
        return results, elapsed

    ids = raw_results["ids"][0]
    docs = raw_results["documents"][0]
    metas = raw_results["metadatas"][0]
    distances = raw_results["distances"][0] if "distances" in raw_results else [0.0] * len(ids)

    for i in range(len(ids)):
        sim_score = _distance_to_similarity(distances[i])
        results.append({
            "candidate_id": ids[i],
            "semantic_similarity_score": round(sim_score, 4),
            "document": docs[i],
            "metadata": metas[i],
        })

    return results, elapsed
