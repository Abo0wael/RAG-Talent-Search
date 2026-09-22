"""
RAG Talent Search Engine - RAG Pipeline

End-to-end Retrieval-Augmented Generation pipeline:
1. Embed recruiter query with all-MiniLM-L6-v2
2. Retrieve Top-K candidates from ChromaDB
3. Select Top 3 candidates for deep LLM evaluation
4. Generate strictly grounded fit explanations with exact resume evidence
5. Candidate-specific Q&A grounded in retrieved resume context
"""

import time
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import TOP_K
from src.retriever import retrieve_candidates
from src.vector_store import get_candidate_by_id, load_collection
from src.llm import get_llm, LLMConfigurationError
from src.prompts import (
    CANDIDATE_EVALUATION_SYSTEM_PROMPT,
    CANDIDATE_EVALUATION_TEMPLATE,
    CANDIDATE_QA_SYSTEM_PROMPT,
    CANDIDATE_QA_TEMPLATE,
)


def _format_candidate_context(candidate: dict[str, Any], rank: int) -> str:
    """Format a single candidate's details for LLM prompt context."""
    meta = candidate.get("metadata", {})
    doc_text = candidate.get("document", "")

    # Clean / truncate document text if extraordinarily long (keep up to 3500 chars)
    trimmed_text = doc_text[:3500] if len(doc_text) > 3500 else doc_text

    return (
        f"--- CANDIDATE #{rank} ---\n"
        f"Candidate ID: {candidate.get('candidate_id')}\n"
        f"Name: {meta.get('name', 'Not specified')}\n"
        f"Designation: {meta.get('designation', 'Not specified')}\n"
        f"Experience: {meta.get('experience', 'Not specified')}\n"
        f"Degree: {meta.get('degree', 'Not specified')}\n"
        f"College: {meta.get('college', 'Not specified')}\n"
        f"Graduation Year: {meta.get('graduation_year', 'Not specified')}\n"
        f"Location: {meta.get('location', 'Not specified')}\n"
        f"Skills Mentioned: {meta.get('skills', 'Not specified')}\n"
        f"Companies: {meta.get('companies', 'Not specified')}\n"
        f"Semantic Similarity Score: {candidate.get('semantic_similarity_score', 'N/A')}\n\n"
        f"Full Resume Content:\n{trimmed_text}\n"
        f"------------------------\n"
    )


class RAGPipeline:
    """End-to-end RAG Talent Search Pipeline."""

    def __init__(self, collection=None):
        self.collection = collection

    def search_and_evaluate(
        self,
        query: str,
        top_k: int = TOP_K,
        num_to_evaluate: int = 3
    ) -> dict[str, Any]:
        """
        Execute full RAG pipeline:
        1. Semantic retrieval of Top-K candidates
        2. LLM evaluation of Top 3 candidates

        Args:
            query: Recruiter query string.
            top_k: Number of candidates to retrieve semantically (default: 5).
            num_to_evaluate: Number of top candidates to evaluate with LLM (default: 3).

        Returns:
            dict containing:
            - query
            - retrieved_candidates (all top_k)
            - top_evaluated (top 3)
            - llm_explanation (markdown string)
            - retrieval_latency (seconds)
            - llm_latency (seconds)
            - total_latency (seconds)
            - error (str | None)
        """
        if not query or not query.strip():
            return {
                "query": query,
                "retrieved_candidates": [],
                "top_evaluated": [],
                "llm_explanation": "Please provide a non-empty search query.",
                "retrieval_latency": 0.0,
                "llm_latency": 0.0,
                "total_latency": 0.0,
                "error": "Empty query"
            }

        start_total = time.perf_counter()

        # Step 1: Semantic Retrieval
        retrieved, retrieval_latency = retrieve_candidates(
            query=query,
            top_k=top_k,
            collection=self.collection
        )

        if not retrieved:
            total_latency = time.perf_counter() - start_total
            return {
                "query": query,
                "retrieved_candidates": [],
                "top_evaluated": [],
                "llm_explanation": "No matching candidates found in the resume database.",
                "retrieval_latency": retrieval_latency,
                "llm_latency": 0.0,
                "total_latency": total_latency,
                "error": None
            }

        # Step 2: Select Top candidates for LLM evaluation
        top_evaluated = retrieved[:num_to_evaluate]

        # Step 3: Format Context for LLM
        context_blocks = [
            _format_candidate_context(candidate, rank=i + 1)
            for i, candidate in enumerate(top_evaluated)
        ]
        combined_context = "\n\n".join(context_blocks)

        prompt_text = CANDIDATE_EVALUATION_TEMPLATE.format(
            query=query.strip(),
            candidates_context=combined_context
        )

        # Step 4: Invoke LLM
        llm_latency = 0.0
        llm_explanation = ""
        llm_error = None

        try:
            llm = get_llm()
            start_llm = time.perf_counter()
            messages = [
                SystemMessage(content=CANDIDATE_EVALUATION_SYSTEM_PROMPT),
                HumanMessage(content=prompt_text),
            ]
            response = llm.invoke(messages)
            llm_latency = time.perf_counter() - start_llm
            llm_explanation = response.content if hasattr(response, "content") else str(response)

        except LLMConfigurationError as e:
            llm_error = str(e)
            llm_explanation = (
                f"⚠️ **LLM Evaluation Unavailable:** {e}\n\n"
                f"Semantic search completed successfully above. To enable AI fit summaries, "
                f"configure your API key in `.env`."
            )
        except Exception as e:
            llm_error = str(e)
            llm_explanation = (
                f"⚠️ **Error during LLM evaluation:** {e}\n\n"
                f"Semantic search results are displayed above."
            )

        total_latency = time.perf_counter() - start_total

        return {
            "query": query,
            "retrieved_candidates": retrieved,
            "top_evaluated": top_evaluated,
            "llm_explanation": llm_explanation,
            "retrieval_latency": round(retrieval_latency, 4),
            "llm_latency": round(llm_latency, 4),
            "total_latency": round(total_latency, 4),
            "error": llm_error
        }

    def ask_candidate_question(
        self,
        candidate_id: str,
        question: str
    ) -> dict[str, Any]:
        """
        Answer a specific question about a candidate grounded solely in their resume text.

        Args:
            candidate_id: Target candidate ID.
            question: Question from the recruiter.

        Returns:
            dict with answer, evidence, latency, and candidate info.
        """
        if not question or not question.strip():
            return {
                "answer": "Please enter a valid question.",
                "candidate_id": candidate_id,
                "latency": 0.0,
                "error": "Empty question"
            }

        candidate = get_candidate_by_id(candidate_id)
        if not candidate:
            return {
                "answer": f"Candidate '{candidate_id}' not found in database.",
                "candidate_id": candidate_id,
                "latency": 0.0,
                "error": "Candidate not found"
            }

        meta = candidate.get("metadata", {})
        candidate_info = f"{meta.get('name', 'Candidate')} - {meta.get('designation', 'Unknown Designation')}"

        prompt_text = CANDIDATE_QA_TEMPLATE.format(
            candidate_id=candidate_id,
            candidate_info=candidate_info,
            resume_text=candidate.get("text", "")[:4000],
            question=question.strip(),
        )

        start = time.perf_counter()
        try:
            llm = get_llm()
            messages = [
                SystemMessage(content=CANDIDATE_QA_SYSTEM_PROMPT),
                HumanMessage(content=prompt_text),
            ]
            response = llm.invoke(messages)
            elapsed = time.perf_counter() - start
            answer = response.content if hasattr(response, "content") else str(response)
            error = None
        except LLMConfigurationError as e:
            elapsed = time.perf_counter() - start
            answer = f"⚠️ LLM not configured: {e}"
            error = str(e)
        except Exception as e:
            elapsed = time.perf_counter() - start
            answer = f"⚠️ Error answering question: {e}"
            error = str(e)

        return {
            "answer": answer,
            "candidate_id": candidate_id,
            "candidate_info": candidate_info,
            "latency": round(elapsed, 4),
            "error": error
        }
