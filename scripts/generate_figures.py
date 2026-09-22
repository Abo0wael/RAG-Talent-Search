"""
Generate and save all visualization figures for the project under outputs/figures/
"""

import sys
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR
from src.data_loader import load_raw_data, get_dataset_stats
from src.resume_parser import parse_all_resumes
from src.retriever import retrieve_candidates


def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    raw_records = load_raw_data()
    stats = get_dataset_stats(raw_records)
    profiles = parse_all_resumes(raw_records)

    # 1. Entity Distribution Plot
    label_df = pd.DataFrame(
        list(stats["label_counts"].items()),
        columns=["Entity Label", "Frequency"]
    )
    plt.figure(figsize=(10, 5))
    plt.barh(label_df["Entity Label"][::-1], label_df["Frequency"][::-1], color="#4F46E5")
    plt.xlabel("Annotation Count")
    plt.title("Entity Frequency in Resume NER Dataset")
    plt.tight_layout()
    fig1 = FIGURES_DIR / "entity_distribution.png"
    plt.savefig(fig1, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig1}")

    # 2. Top Skills Frequency
    all_skills = []
    for p in profiles:
        all_skills.extend(p.get("skills", []))
    top_skills = Counter(all_skills).most_common(20)
    skills_df = pd.DataFrame(top_skills, columns=["Skill", "Count"])

    plt.figure(figsize=(10, 6))
    plt.barh(skills_df["Skill"][::-1], skills_df["Count"][::-1], color="#06B6D4")
    plt.xlabel("Frequency across Resumes")
    plt.title("Top 20 Most Frequent Skills in Resumes")
    plt.tight_layout()
    fig2 = FIGURES_DIR / "skills_frequency.png"
    plt.savefig(fig2, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig2}")

    # 3. Top Designations
    all_designations = [p.get("designation") for p in profiles if p.get("designation")]
    top_desigs = Counter(all_designations).most_common(15)
    desig_df = pd.DataFrame(top_desigs, columns=["Designation", "Count"])

    plt.figure(figsize=(10, 5))
    plt.barh(desig_df["Designation"][::-1], desig_df["Count"][::-1], color="#10B981")
    plt.xlabel("Count")
    plt.title("Top 15 Most Common Job Designations")
    plt.tight_layout()
    fig3 = FIGURES_DIR / "designation_distribution.png"
    plt.savefig(fig3, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig3}")

    # 4. Retrieval Similarity Scores for Sample Query
    query = "Find me a Junior Data Analyst who knows SQL and Tableau."
    results, latency = retrieve_candidates(query, top_k=8)
    scores = [r["semantic_similarity_score"] for r in results]
    names = [r["metadata"].get("name", f"Candidate {i+1}") for i, r in enumerate(results)]

    plt.figure(figsize=(10, 4.5))
    plt.bar(names, scores, color="#8B5CF6")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Semantic Similarity Score")
    plt.title(f"Semantic Similarity Scores: '{query[:40]}...'")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    fig4 = FIGURES_DIR / "retrieval_similarity_scores.png"
    plt.savefig(fig4, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig4}")

    # 5. Latency Comparison across 5 benchmark queries
    test_queries = [
        "Junior Data Analyst (SQL, Tableau)",
        "Senior Java Developer (Spring, Hibernate)",
        "QA Automation Tester (Selenium, Python)",
        "Database Administrator (Oracle, SQL)",
        "Front-end Developer (HTML, CSS, JS)"
    ]
    latencies = []
    for q in test_queries:
        _, lat = retrieve_candidates(q, top_k=5)
        latencies.append(lat * 1000)  # in ms

    plt.figure(figsize=(10, 4))
    plt.barh(test_queries[::-1], latencies[::-1], color="#EC4899")
    plt.xlabel("Retrieval Latency (milliseconds)")
    plt.title("ChromaDB Vector Retrieval Latency per Query")
    plt.tight_layout()
    fig5 = FIGURES_DIR / "retrieval_latency.png"
    plt.savefig(fig5, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig5}")


if __name__ == "__main__":
    generate_all_figures()
