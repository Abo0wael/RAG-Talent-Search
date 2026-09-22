"""
RAG Talent Search Engine - Evaluation Module

Evaluates the retrieval and generation pipeline using an attribute-level
ground-truth relevance methodology.

METHODOLOGY RULES:
1. No fabricated candidate-level ground-truth labels.
2. Evaluates Attribute-Level Retrieval Relevance against documented recruiter queries.
3. Computes Precision@K and Recall@K based on verifiable presence of expected job attributes.
4. MRR is ONLY computed when explicit candidate IDs are provided (otherwise documented as N/A).
5. Measures retrieval and LLM response latencies.
"""

import json
import time
from pathlib import Path
from typing import Any

from src.config import EVALUATION_DIR, TOP_K
from src.retriever import retrieve_candidates
from src.vector_store import get_all_candidates, load_collection
from src.llm import get_llm, LLMConfigurationError
from src.prompts import CANDIDATE_EVALUATION_SYSTEM_PROMPT, CANDIDATE_EVALUATION_TEMPLATE
from langchain_core.messages import SystemMessage, HumanMessage


# Documented Benchmark Queries with explicitly expected job attributes
BENCHMARK_QUERIES = [
    {
        "query_id": "Q1",
        "query": "Junior Data Analyst who knows SQL and Tableau",
        "expected_attributes": ["data analyst", "sql", "tableau"],
        "category": "Data & Analytics",
        "relevant_candidate_ids": None  # No pre-assigned candidate IDs
    },
    {
        "query_id": "Q2",
        "query": "Senior Java Developer with Spring and Hibernate frameworks",
        "expected_attributes": ["java", "spring", "hibernate", "developer"],
        "category": "Software Engineering",
        "relevant_candidate_ids": None
    },
    {
        "query_id": "Q3",
        "query": "QA Automation Engineer experienced in Selenium and Python",
        "expected_attributes": ["qa", "testing", "selenium", "python"],
        "category": "Quality Assurance",
        "relevant_candidate_ids": None
    },
    {
        "query_id": "Q4",
        "query": "Database Administrator with Oracle and SQL Server experience",
        "expected_attributes": ["dba", "database administrator", "oracle", "sql"],
        "category": "Database Administration",
        "relevant_candidate_ids": None
    },
    {
        "query_id": "Q5",
        "query": "Front-end Developer proficient in HTML, CSS, JavaScript, and Angular",
        "expected_attributes": ["html", "css", "javascript", "angular"],
        "category": "Web Development",
        "relevant_candidate_ids": None
    }
]


def candidate_matches_attributes(candidate: dict[str, Any], attributes: list[str]) -> tuple[bool, list[str]]:
    """
    Check if a candidate document or metadata contains expected job attributes.

    Returns:
        tuple of (is_relevant, matched_attributes_list)
    """
    text = (candidate.get("document", "") or candidate.get("text", "")).lower()
    meta = candidate.get("metadata", {})
    skills = str(meta.get("skills", "")).lower()
    designation = str(meta.get("designation", "")).lower()

    combined_search_space = f"{text} {skills} {designation}"

    matched = []
    for attr in attributes:
        if attr.lower() in combined_search_space:
            matched.append(attr)

    # Candidate is considered attribute-relevant if at least 1 core attribute matches
    is_relevant = len(matched) > 0
    return is_relevant, matched


def compute_attribute_precision_recall(
    retrieved_candidates: list[dict[str, Any]],
    all_candidates: list[dict[str, Any]],
    expected_attributes: list[str],
    k: int
) -> dict[str, Any]:
    """
    Compute Precision@K and Recall@K based on verified attribute presence.

    Precision@K = (# of retrieved Top-K candidates containing expected attributes) / K
    Recall@K = (# of retrieved relevant candidates in Top-K) / (Total relevant candidates in entire DB)
    """
    top_k_candidates = retrieved_candidates[:k]

    # Evaluate Top-K retrieved
    retrieved_relevant_count = 0
    retrieved_details = []

    for c in top_k_candidates:
        is_rel, matched = candidate_matches_attributes(c, expected_attributes)
        if is_rel:
            retrieved_relevant_count += 1
        retrieved_details.append({
            "candidate_id": c.get("candidate_id"),
            "is_relevant": is_rel,
            "matched_attributes": matched,
            "score": c.get("semantic_similarity_score", 0.0)
        })

    precision_at_k = retrieved_relevant_count / max(k, 1)

    # Compute Total relevant in entire dataset for true Recall@K
    total_relevant_in_db = 0
    for c in all_candidates:
        is_rel, _ = candidate_matches_attributes(c, expected_attributes)
        if is_rel:
            total_relevant_in_db += 1

    recall_at_k = (
        retrieved_relevant_count / total_relevant_in_db
        if total_relevant_in_db > 0 else 0.0
    )

    return {
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "retrieved_relevant_count": retrieved_relevant_count,
        "total_relevant_in_db": total_relevant_in_db,
        "k": k,
        "retrieved_details": retrieved_details
    }


def run_evaluation(
    queries: list[dict] | None = None,
    k: int = TOP_K,
    evaluate_llm_latency: bool = True
) -> dict[str, Any]:
    """
    Run full retrieval evaluation across benchmark queries.

    Args:
        queries: Optional list of benchmark queries. Defaults to BENCHMARK_QUERIES.
        k: Top-K rank cutoff.
        evaluate_llm_latency: Whether to benchmark LLM inference on the first query.

    Returns:
        dict with overall metrics, per-query breakdowns, and methodology notes.
    """
    eval_queries = queries or BENCHMARK_QUERIES
    collection = load_collection()
    all_candidates = get_all_candidates()

    per_query_results = []
    total_retrieval_latencies = []
    precisions = []
    recalls = []

    print(f"[Evaluation] Evaluating {len(eval_queries)} queries against {len(all_candidates)} candidates...")

    for q in eval_queries:
        query_text = q["query"]
        attrs = q["expected_attributes"]
        relevant_ids = q.get("relevant_candidate_ids")

        # 1. Measure Retrieval
        retrieved, ret_latency = retrieve_candidates(query_text, top_k=k, collection=collection)
        total_retrieval_latencies.append(ret_latency)

        # 2. Compute Precision@K and Recall@K
        pr_metrics = compute_attribute_precision_recall(
            retrieved_candidates=retrieved,
            all_candidates=all_candidates,
            expected_attributes=attrs,
            k=k
        )

        precisions.append(pr_metrics["precision_at_k"])
        recalls.append(pr_metrics["recall_at_k"])

        # 3. Explicit MRR calculation check
        if relevant_ids is not None and len(relevant_ids) > 0:
            # Only compute MRR if explicit ground-truth candidate IDs exist
            mrr = 0.0
            for rank, c in enumerate(retrieved[:k], 1):
                if c["candidate_id"] in relevant_ids:
                    mrr = 1.0 / rank
                    break
        else:
            # Do NOT fabricate ground truth
            mrr = None

        per_query_results.append({
            "query_id": q.get("query_id"),
            "query": query_text,
            "category": q.get("category"),
            "expected_attributes": attrs,
            "precision_at_k": pr_metrics["precision_at_k"],
            "recall_at_k": pr_metrics["recall_at_k"],
            "mrr": mrr,
            "mrr_note": "N/A (Candidate ground truth not provided; fabrication omitted)" if mrr is None else "Calculated",
            "retrieval_latency_sec": round(ret_latency, 4),
            "relevant_retrieved": pr_metrics["retrieved_relevant_count"],
            "total_relevant_in_db": pr_metrics["total_relevant_in_db"],
        })

    # Benchmark LLM latency on sample query
    llm_sample_latency = None
    if evaluate_llm_latency:
        try:
            llm = get_llm()
            sample_query = eval_queries[0]["query"]
            sample_prompt = f"Summarize candidate match for: {sample_query}"
            start_llm = time.perf_counter()
            _ = llm.invoke([HumanMessage(content=sample_prompt)])
            llm_sample_latency = round(time.perf_counter() - start_llm, 4)
        except Exception as e:
            llm_sample_latency = f"Unavailable: {e}"

    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_ret_latency = sum(total_retrieval_latencies) / len(total_retrieval_latencies) if total_retrieval_latencies else 0.0

    summary = {
        "evaluation_methodology": "Attribute-Level Retrieval Relevance (No fabricated candidate IDs)",
        "num_queries_evaluated": len(eval_queries),
        "top_k": k,
        "mean_precision_at_k": round(avg_precision, 4),
        "mean_recall_at_k": round(avg_recall, 4),
        "mrr_status": "Omitted per design specification (no ground-truth candidate IDs)",
        "average_retrieval_latency_sec": round(avg_ret_latency, 4),
        "sample_llm_latency_sec": llm_sample_latency,
        "query_results": per_query_results
    }

    # Save to outputs/evaluation
    Path(EVALUATION_DIR).mkdir(parents=True, exist_ok=True)
    out_file = Path(EVALUATION_DIR) / "evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Evaluation] Evaluation completed. Results saved to {out_file}")
    return summary
