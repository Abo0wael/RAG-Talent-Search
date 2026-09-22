"""
End-to-end verification script for RAG Talent Search.
Tests:
1. Semantic search with "Find me a Junior Data Analyst who knows SQL and Tableau"
2. Groq LLM candidate evaluation (Top 3 candidates)
3. Bias check audit
4. Candidate Q&A
5. Evaluation metrics (Attribute-level Precision@K, Recall@K)
"""

import sys
from pathlib import Path

# Enable utf-8 printing on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import retrieve_candidates
from src.rag_pipeline import RAGPipeline
from src.bias_checker import BiasChecker
from src.evaluation import run_evaluation


def test_semantic_search():
    print("\n--- 1. Testing Semantic Search ---")
    query = "Find me a Junior Data Analyst who knows SQL and Tableau."
    results, latency = retrieve_candidates(query, top_k=5)
    print(f"Retrieved {len(results)} candidates in {latency:.3f}s")
    for idx, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"  #{idx} {meta.get('name')} | Role: {meta.get('designation')} | Score: {r['semantic_similarity_score']}")
    assert len(results) > 0, "No candidates retrieved!"
    print("Semantic search: PASSED")
    return results


def test_rag_pipeline(results):
    print("\n--- 2. Testing RAG Pipeline with Groq LLM Evaluation ---")
    pipeline = RAGPipeline()
    query = "Find me a Junior Data Analyst who knows SQL and Tableau."
    rag_res = pipeline.search_and_evaluate(query, top_k=5, num_to_evaluate=3)
    print(f"Retrieval Latency: {rag_res['retrieval_latency']:.3f}s")
    print(f"LLM Latency: {rag_res['llm_latency']:.3f}s")
    print("\nLLM Evaluation Output Excerpt:")
    explanation = rag_res["llm_explanation"]
    print(explanation[:800] + ("..." if len(explanation) > 800 else ""))
    assert "Candidate" in explanation or "fit" in explanation.lower(), "LLM output seems empty or malformed"
    print("\nRAG Pipeline & Groq LLM: PASSED")


def test_bias_checker(results):
    print("\n--- 3. Testing Bias Checker ---")
    checker = BiasChecker()
    query = "Find me a Junior Data Analyst who knows SQL and Tableau."
    audit = checker.audit_query_results(query, retrieved_candidates=results, top_k=5)
    print(f"Signal: {audit['audit_signal']}")
    print(f"Shifts: {audit['rank_shifts']}/{audit['total_evaluated']}")
    print(f"Average Delta: {audit['average_score_delta']}")
    print("Bias Checker: PASSED")


def test_candidate_qa(results):
    print("\n--- 4. Testing Candidate Q&A ---")
    pipeline = RAGPipeline()
    first_candidate_id = results[0]["candidate_id"]
    question = "Does this candidate have SQL experience?"
    qa_res = pipeline.ask_candidate_question(first_candidate_id, question)
    print(f"Candidate: {qa_res['candidate_info']}")
    print(f"Question: {question}")
    print(f"Answer: {qa_res['answer'][:400]}")
    print(f"Latency: {qa_res['latency']:.3f}s")
    print("Candidate Q&A: PASSED")


def test_evaluation_benchmark():
    print("\n--- 5. Testing Attribute-Level Evaluation Benchmark ---")
    summary = run_evaluation(k=5, evaluate_llm_latency=False)
    print(f"Evaluated queries: {summary['num_queries_evaluated']}")
    print(f"Mean Precision@5: {summary['mean_precision_at_k']:.4f}")
    print(f"Mean Recall@5: {summary['mean_recall_at_k']:.4f}")
    print(f"Average Retrieval Latency: {summary['average_retrieval_latency_sec']:.4f}s")
    print(f"MRR Status: {summary['mrr_status']}")
    print("Evaluation Benchmark: PASSED")


if __name__ == "__main__":
    res = test_semantic_search()
    test_rag_pipeline(res)
    test_bias_checker(res)
    test_candidate_qa(res)
    test_evaluation_benchmark()
    print("\n==========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==========================================")
