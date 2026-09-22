"""
RAG Talent Search Engine - Bias Checker

Implements an explainable bias-audit mechanism that assesses whether candidate
retrieval and ranking may be influenced by non-job-relevant or sensitive metadata.

STRICT FAIRNESS CONSTRAINTS:
1. NEVER infers sensitive demographic attributes (gender, race, ethnicity, religion).
2. NEVER fabricates demographic labels or classifications.
3. Uses observable, explainable ranking comparison:
   A. Ranking using full candidate text (with names, locations, colleges, graduation years)
   vs.
   B. Ranking using sanitized, job-relevant candidate text (non-job-relevant entities masked)
4. Produces informational audit signals with explicit limitations disclosures.
"""

import re
from typing import Any

from src.config import SENSITIVE_FIELDS, JOB_RELEVANT_FIELDS
from src.embeddings import get_embedding_model
from src.retriever import retrieve_candidates
from src.vector_store import load_collection
from src.llm import get_llm, LLMConfigurationError


def sanitize_resume_text(text: str, metadata: dict[str, Any]) -> str:
    """
    Produce a sanitized, job-relevant version of resume text by masking out
    potentially sensitive or non-job-relevant entity values.

    Masks:
    - Name -> [REDACTED_NAME]
    - Location -> [REDACTED_LOCATION]
    - College -> [REDACTED_COLLEGE]
    - Graduation Year -> [REDACTED_YEAR]
    - Email -> [REDACTED_EMAIL]
    """
    sanitized = text

    # Mask email
    email = metadata.get("email")
    if email and email != "Not specified":
        sanitized = re.sub(re.escape(str(email).strip()), "[REDACTED_EMAIL]", sanitized, flags=re.IGNORECASE)

    # Mask name
    name = metadata.get("name")
    if name and name != "Not specified" and len(str(name).strip()) > 2:
        sanitized = re.sub(re.escape(str(name).strip()), "[REDACTED_NAME]", sanitized, flags=re.IGNORECASE)

    # Mask college
    college = metadata.get("college")
    if college and college != "Not specified" and len(str(college).strip()) > 3:
        sanitized = re.sub(re.escape(str(college).strip()), "[REDACTED_COLLEGE]", sanitized, flags=re.IGNORECASE)

    # Mask location
    location = metadata.get("location")
    if location and location != "Not specified" and len(str(location).strip()) > 2:
        sanitized = re.sub(re.escape(str(location).strip()), "[REDACTED_LOCATION]", sanitized, flags=re.IGNORECASE)

    # Mask graduation year (4-digit years like 2012, 2016)
    grad_year = metadata.get("graduation_year")
    if grad_year and grad_year != "Not specified":
        match = re.search(r'\b(19\d\d|20\d\d)\b', str(grad_year))
        if match:
            year = match.group(1)
            sanitized = re.sub(r'\b' + year + r'\b', "[REDACTED_YEAR]", sanitized)

    return sanitized


def compute_sanitized_similarity(query: str, sanitized_text: str) -> float:
    """Compute cosine similarity between query and sanitized text vector."""
    model = get_embedding_model()
    # Embeddings are normalized by our wrapper
    q_vec = model.embed_query(query)
    d_vec = model.embed_query(sanitized_text)

    # Dot product of normalized vectors = cosine similarity
    dot = sum(a * b for a, b in zip(q_vec, d_vec))
    return max(0.0, min(1.0, float(dot)))


class BiasChecker:
    """Audit mechanism for evaluating non-job-relevant attribute sensitivity."""

    def __init__(self, collection=None):
        self.collection = collection

    def audit_query_results(
        self,
        query: str,
        retrieved_candidates: list[dict[str, Any]] | None = None,
        top_k: int = 5
    ) -> dict[str, Any]:
        """
        Run a comparative bias audit on the query retrieval.

        Compares:
        A. Full-text retrieval ranking
        B. Sanitized job-relevant ranking (sensitive metadata masked)

        Returns:
            dict containing:
            - audit_signal: Warning or Neutral status string
            - ranking_comparison: list comparing ranks and score deltas
            - rank_shifts: count and details of rank changes
            - sensitive_attributes_present: count and breakdown
            - audit_summary: narrative explanation
            - limitations: mandatory legal and technical disclaimer
        """
        # If candidates not provided, retrieve them
        if retrieved_candidates is None:
            retrieved_candidates, _ = retrieve_candidates(
                query=query,
                top_k=top_k,
                collection=self.collection
            )

        if not retrieved_candidates:
            return {
                "audit_signal": "No candidates to evaluate",
                "ranking_comparison": [],
                "rank_shifts": 0,
                "sensitive_attributes_present": {},
                "audit_summary": "No candidate resumes were retrieved for this query.",
                "limitations": self.get_limitations_statement()
            }

        # Step 1: Baseline (Full Metadata) Ranking
        baseline_candidates = retrieved_candidates[:top_k]

        # Step 2: Compute Sanitized Similarity for these candidates
        sanitized_scores = []
        sensitive_counts = {field: 0 for field in SENSITIVE_FIELDS}

        for c in baseline_candidates:
            doc = c.get("document", "")
            meta = c.get("metadata", {})

            # Count sensitive attributes present in profile
            for field in SENSITIVE_FIELDS:
                val = meta.get(field)
                if val and val != "Not specified" and str(val).strip():
                    sensitive_counts[field] += 1

            # Create sanitized text
            sanitized_text = sanitize_resume_text(doc, meta)
            sim = compute_sanitized_similarity(query, sanitized_text)

            sanitized_scores.append({
                "candidate_id": c.get("candidate_id"),
                "name": meta.get("name", "Candidate"),
                "original_score": c.get("semantic_similarity_score", 0.0),
                "sanitized_score": round(sim, 4),
                "delta": round(abs(c.get("semantic_similarity_score", 0.0) - sim), 4),
            })

        # Step 3: Rank by sanitized score
        sanitized_ranked = sorted(sanitized_scores, key=lambda x: -x["sanitized_score"])

        # Determine rank positions
        baseline_order = [c["candidate_id"] for c in baseline_candidates]
        sanitized_order = [c["candidate_id"] for c in sanitized_ranked]

        rank_shifts = 0
        comparison_table = []

        for original_rank, c in enumerate(baseline_candidates, 1):
            cid = c["candidate_id"]
            sanitized_rank = sanitized_order.index(cid) + 1
            shift = abs(original_rank - sanitized_rank)
            if shift > 0:
                rank_shifts += 1

            sanitized_info = next(item for item in sanitized_scores if item["candidate_id"] == cid)

            comparison_table.append({
                "candidate_id": cid,
                "name": c.get("metadata", {}).get("name", "Candidate"),
                "baseline_rank": original_rank,
                "sanitized_rank": sanitized_rank,
                "rank_shift": shift,
                "baseline_score": sanitized_info["original_score"],
                "sanitized_score": sanitized_info["sanitized_score"],
                "score_delta": sanitized_info["delta"],
                "location": c.get("metadata", {}).get("location", "Not specified"),
                "college": c.get("metadata", {}).get("college", "Not specified"),
                "graduation_year": c.get("metadata", {}).get("graduation_year", "Not specified"),
            })

        # Step 4: Determine Signal
        # A significant shift is defined as rank shift in >= 40% of candidates or average score delta > 0.05
        avg_delta = sum(item["score_delta"] for item in comparison_table) / max(len(comparison_table), 1)

        if rank_shifts >= 2 or avg_delta > 0.05:
            audit_signal = "⚠️ Potential non-job-relevant attribute influence detected."
            signal_level = "warning"
        else:
            audit_signal = "✅ Low non-job-relevant attribute sensitivity observed."
            signal_level = "normal"

        # Step 5: Narrative Summary
        audit_summary = (
            f"Evaluated {len(comparison_table)} candidates. "
            f"{rank_shifts} of {len(comparison_table)} candidate ranks shifted when non-job-relevant metadata "
            f"(name, location, college, graduation year) was masked. "
            f"Average semantic score divergence: {avg_delta:.4f}. "
            f"Audit signal: {audit_signal}"
        )

        return {
            "audit_signal": audit_signal,
            "signal_level": signal_level,
            "ranking_comparison": comparison_table,
            "rank_shifts": rank_shifts,
            "total_evaluated": len(comparison_table),
            "average_score_delta": round(avg_delta, 4),
            "sensitive_attributes_present": sensitive_counts,
            "audit_summary": audit_summary,
            "limitations": self.get_limitations_statement()
        }

    @staticmethod
    def get_limitations_statement() -> str:
        """Return clear, explicit limitations disclosure for this audit."""
        return (
            "AUDIT LIMITATIONS & DISCLAIMER:\n"
            "1. Informational Signal Only: This audit compares vector similarity deltas when "
            "non-job-relevant text (names, locations, universities, graduation years) is masked. "
            "It does NOT constitute proof of bias or legal non-compliance.\n"
            "2. No Demographic Inference: The system does not infer protected classes (race, gender, "
            "ethnicity, religion, sexual orientation) from candidate names, locations, or schools.\n"
            "3. Lexical Co-occurrence: Pretrained language models may retain latent associations between "
            "technical terminology, institutions, or regions that simple keyword masking cannot fully isolate.\n"
            "4. Human Oversight Required: Algorithmic screening must always be supervised by human "
            "recruiters adhering to standardized, objective job criteria."
        )
