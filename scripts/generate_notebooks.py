"""
Script to generate standard Jupyter notebooks for the project.
Creates:
  notebooks/01_data_exploration.ipynb
  notebooks/02_resume_processing.ipynb
  notebooks/03_embeddings_and_vector_db.ipynb
  notebooks/04_semantic_search.ipynb
  notebooks/05_rag_evaluation.ipynb
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def markdown_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


# =============================================================================
# Notebook 1: Data Exploration
# =============================================================================
nb1_cells = [
    markdown_cell("# 01 - Resume NER Dataset Exploration\n\nThis notebook inspects the raw DataTurks Resume Entities NER dataset, calculates summary statistics, and visualizes entity distributions."),
    code_cell("""import sys
from pathlib import Path

# Setup project path
PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import load_raw_data, get_dataset_stats
from src.config import FIGURES_DIR"""),
    markdown_cell("## 1. Load Raw Dataset"),
    code_cell("""raw_records = load_raw_data()
stats = get_dataset_stats(raw_records)

print(f"Total Resumes: {stats['total_resumes']}")
print(f"Average Text Length: {stats['avg_text_length']:.1f} characters")
print(f"Min Text Length: {stats['min_text_length']} characters")
print(f"Max Text Length: {stats['max_text_length']} characters")"""),
    markdown_cell("## 2. Inspect Entity Label Distribution"),
    code_cell("""label_df = pd.DataFrame(
    list(stats["label_counts"].items()),
    columns=["Entity Label", "Frequency"]
)
print(label_df)"""),
    markdown_cell("## 3. Visualize & Save Entity Distribution Chart"),
    code_cell("""FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.barh(label_df["Entity Label"][::-1], label_df["Frequency"][::-1], color="#4F46E5")
plt.xlabel("Annotation Count")
plt.title("Entity Frequency in Resume NER Dataset")
plt.tight_layout()

figure_path = FIGURES_DIR / "entity_distribution.png"
plt.savefig(figure_path, dpi=300)
print(f"Figure saved to: {figure_path}")
plt.show()"""),
    markdown_cell("## 4. Inspect Sample Record Structure"),
    code_cell("""import json
sample = raw_records[0]
print("Resume Text Preview:")
print(sample["content"][:400] + "...")
print("\\nFirst 3 Annotations:")
print(json.dumps(sample["annotation"][:3], indent=2))""")
]

# =============================================================================
# Notebook 2: Resume Processing
# =============================================================================
nb2_cells = [
    markdown_cell("# 02 - Resume Parsing and Normalization\n\nParses DataTurks NER annotations into structured candidate profiles without fabricating missing attributes."),
    code_cell("""import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data_loader import load_raw_data
from src.resume_parser import parse_resume, parse_all_resumes"""),
    markdown_cell("## 1. Parse All Resumes"),
    code_cell("""raw_records = load_raw_data()
profiles = parse_all_resumes(raw_records)
print(f"Parsed {len(profiles)} normalized candidate profiles.")"""),
    markdown_cell("## 2. Inspect Sample Candidate Profile"),
    code_cell("""sample_profile = profiles[0]
for k, v in sample_profile.items():
    if k != "text":
        print(f"{k}: {v}")"""),
    markdown_cell("## 3. Top Skills Frequency Analysis"),
    code_cell("""from collections import Counter
import matplotlib.pyplot as plt
from src.config import FIGURES_DIR

all_skills = []
for p in profiles:
    all_skills.extend(p.get("skills", []))

skill_counts = Counter(all_skills).most_common(20)
skills_df = pd.DataFrame(skill_counts, columns=["Skill", "Count"])

plt.figure(figsize=(10, 6))
plt.barh(skills_df["Skill"][::-1], skills_df["Count"][::-1], color="#06B6D4")
plt.xlabel("Occurrences across Resumes")
plt.title("Top 20 Most Frequent Skills in Resumes")
plt.tight_layout()

skills_fig_path = FIGURES_DIR / "top_skills_frequency.png"
plt.savefig(skills_fig_path, dpi=300)
print(f"Saved skills figure to: {skills_fig_path}")
plt.show()""")
]

# =============================================================================
# Notebook 3: Embeddings & Vector DB
# =============================================================================
nb3_cells = [
    markdown_cell("# 03 - Local Embeddings and ChromaDB Indexing\n\nGenerates dense vector embeddings using `all-MiniLM-L6-v2` and persists complete resume documents into ChromaDB."),
    code_cell("""import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_data
from src.resume_parser import parse_all_resumes
from src.embeddings import get_embedding_model, embed_query
from src.vector_store import build_index, load_collection, get_collection_stats"""),
    markdown_cell("## 1. Load Local Embedding Model"),
    code_cell("""embedder = get_embedding_model()
test_vec = embed_query("Data Analyst with SQL")
print(f"Embedding dimension: {len(test_vec)}")
print(f"First 5 vector components: {test_vec[:5]}")"""),
    markdown_cell("## 2. Build or Load ChromaDB Index"),
    code_cell("""raw_records = load_raw_data()
profiles = parse_all_resumes(raw_records)

# Build or reuse persistent index
doc_count = build_index(profiles, rebuild=False)
stats = get_collection_stats()
print(f"Collection status: {stats}")"""),
    markdown_cell("## 3. Verify Document Retrieval"),
    code_cell("""collection = load_collection()
sample_query_res = collection.query(
    query_embeddings=[test_vec],
    n_results=2,
    include=["metadatas", "distances"]
)
print("Top match metadata:")
print(sample_query_res["metadatas"][0][0])
print(f"Cosine distance: {sample_query_res['distances'][0][0]:.4f}")""")
]

# =============================================================================
# Notebook 4: Semantic Search
# =============================================================================
nb4_cells = [
    markdown_cell("# 04 - Semantic Search Exploration\n\nEvaluates natural-language recruiter queries, analyzes similarity score distributions, and inspects retrieved candidates."),
    code_cell("""import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from src.retriever import retrieve_candidates
from src.config import FIGURES_DIR"""),
    markdown_cell("## 1. Execute Sample Recruiter Queries"),
    code_cell("""test_queries = [
    "Find me a Junior Data Analyst who knows SQL and Tableau.",
    "Senior Java Developer with Spring and Hibernate.",
    "QA Tester experienced in automation and Selenium."
]

for q in test_queries:
    results, latency = retrieve_candidates(q, top_k=3)
    print(f"\\nQuery: {q} (Latency: {latency:.4f}s)")
    for rank, r in enumerate(results, 1):
        meta = r['metadata']
        print(f"  #{rank}: {meta.get('name')} | {meta.get('designation')} | Sim: {r['semantic_similarity_score']}")"""),
    markdown_cell("## 2. Visualize Similarity Score Distribution"),
    code_cell("""primary_query = "Find me a Junior Data Analyst who knows SQL and Tableau."
top_res, _ = retrieve_candidates(primary_query, top_k=10)

scores = [r["semantic_similarity_score"] for r in top_res]
names = [r["metadata"].get("name", f"Candidate {i+1}") for i, r in enumerate(top_res)]

plt.figure(figsize=(9, 4))
plt.bar(names, scores, color="#6366F1")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Semantic Similarity Score")
plt.title(f"Similarity Scores for: '{primary_query[:35]}...'")
plt.tight_layout()

sim_fig_path = FIGURES_DIR / "sample_similarity_scores.png"
plt.savefig(sim_fig_path, dpi=300)
print(f"Saved figure: {sim_fig_path}")
plt.show()""")
]

# =============================================================================
# Notebook 5: RAG Evaluation
# =============================================================================
nb5_cells = [
    markdown_cell("# 05 - RAG Pipeline & Retrieval Evaluation\n\nRuns attribute-level retrieval evaluation (Precision@K, Recall@K) without fabricated candidate labels, tests Groq LLM candidate fit explanations, and benchmarks latency."),
    code_cell("""import sys
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.evaluation import run_evaluation
from src.rag_pipeline import RAGPipeline"""),
    markdown_cell("## 1. Run Attribute-Level Retrieval Evaluation"),
    code_cell("""eval_summary = run_evaluation(k=5, evaluate_llm_latency=True)
print(f"Mean Precision@5: {eval_summary['mean_precision_at_k']:.4f}")
print(f"Mean Recall@5: {eval_summary['mean_recall_at_k']:.4f}")
print(f"Average Retrieval Latency: {eval_summary['average_retrieval_latency_sec']:.4f}s")
print(f"Sample LLM Latency: {eval_summary['sample_llm_latency_sec']}s")
print(f"MRR Status: {eval_summary['mrr_status']}")"""),
    markdown_cell("## 2. Detailed Per-Query Results"),
    code_cell("""df_eval = pd.DataFrame(eval_summary["query_results"])
display_cols = ["query_id", "query", "precision_at_k", "recall_at_k", "retrieval_latency_sec", "mrr_note"]
print(df_eval[display_cols])"""),
    markdown_cell("## 3. End-to-End RAG Pipeline Demo (Semantic Search + LLM Explanation)"),
    code_cell("""pipeline = RAGPipeline()
query = "Find me a Junior Data Analyst who knows SQL and Tableau."
rag_result = pipeline.search_and_evaluate(query, top_k=3, num_to_evaluate=3)

print("Retrieval Latency:", rag_result["retrieval_latency"], "s")
print("LLM Latency:", rag_result["llm_latency"], "s")
print("\\n--- LLM CANDIDATE EVALUATION OUTPUT ---\\n")
print(rag_result["llm_explanation"])""")
]

# Write out notebooks
notebooks = {
    "01_data_exploration.ipynb": nb1_cells,
    "02_resume_processing.ipynb": nb2_cells,
    "03_embeddings_and_vector_db.ipynb": nb3_cells,
    "04_semantic_search.ipynb": nb4_cells,
    "05_rag_evaluation.ipynb": nb5_cells,
}

for filename, cells in notebooks.items():
    nb_path = NOTEBOOKS_DIR / filename
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=1)
    print(f"Generated: {nb_path}")

print("All 5 notebooks generated successfully.")
