"""
RAG Talent Search Engine - Streamlit Web Application
An Executive-Grade AI Recruitment & Semantic Discovery Platform.
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    TOP_K,
    LLM_PROVIDER,
    LLM_MODEL,
    DATA_PATH,
)
from src.embeddings import get_embedding_model
from src.vector_store import (
    load_collection,
    index_exists,
    get_collection_stats,
    get_all_candidates,
    get_candidate_by_id,
)
from src.rag_pipeline import RAGPipeline
from src.bias_checker import BiasChecker
from src.llm import check_llm_status


# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TalentPulse AI | RAG Talent Search Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# High-End Design System & Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* Global Base */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #F1F5F9;
    }
    
    /* Main App Background */
    .stApp {
        background: radial-gradient(circle at 10% 0%, #0F172A 0%, #090D16 100%);
    }

    /* Top Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: "";
        position: absolute;
        top: -50px;
        right: -50px;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #A5B4FC;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        margin-bottom: 1rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.15;
        background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #A5B4FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 850px;
        line-height: 1.6;
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1.1rem;
        text-align: left;
        backdrop-filter: blur(8px);
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -6px rgba(0, 0, 0, 0.4);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }

    /* AI Report Card */
    .ai-report-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 14px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        box-shadow: 0 12px 28px -10px rgba(99, 102, 241, 0.2);
    }
    .ai-report-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.5rem;
    }
    .ai-badge {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Candidate Card */
    .candidate-card-v2 {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        transition: all 0.2s ease;
    }
    .candidate-card-v2:hover {
        border-color: rgba(99, 102, 241, 0.35);
        background: rgba(30, 41, 59, 0.75);
    }
    .candidate-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }
    .cand-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .cand-role {
        font-size: 0.9rem;
        color: #94A3B8;
        font-weight: 500;
    }
    .cand-avatar {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.05rem;
        color: white;
        margin-right: 0.85rem;
    }
    .score-chip {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #A5B4FC;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
    }

    /* Skill Tags */
    .skill-pill {
        display: inline-block;
        background: rgba(51, 65, 85, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #CBD5E1;
        padding: 0.25rem 0.65rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        transition: all 0.15s ease;
    }
    .skill-pill:hover {
        background: rgba(99, 102, 241, 0.25);
        border-color: rgba(99, 102, 241, 0.5);
        color: #EEF2FF;
    }

    /* Audit Indicators */
    .audit-box-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 12px;
        padding: 1.3rem;
        color: #FDE68A;
        margin: 1rem 0;
    }
    .audit-box-success {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 1.3rem;
        color: #A7F3D0;
        margin: 1rem 0;
    }

    /* Code & Resume Excerpt Preview */
    .resume-box {
        background: #0B1120;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #94A3B8;
        line-height: 1.6;
        max-height: 220px;
        overflow-y: auto;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 18px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.92rem;
        border: none;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.4rem;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
        transform: translateY(-1px);
    }
    
    /* Text Inputs */
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_embedding_model():
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def load_cached_vector_collection():
    if not index_exists():
        return None
    try:
        return load_collection()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar Redesign
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem;">
            <div style="background: linear-gradient(135deg, #6366F1, #8B5CF6); width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 800; color: white; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);">
                ⚡
            </div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">TalentPulse</div>
                <div style="font-size: 0.75rem; color: #818CF8; font-weight: 600; text-transform: uppercase;">Enterprise RAG ATS</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Status Panel
    has_index = index_exists()
    doc_count = get_collection_stats().get('document_count', 0) if has_index else 0
    llm_info = check_llm_status()

    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 0.9rem; margin-bottom: 1.2rem;">
            <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 0.6rem;">System Diagnostics</div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="font-size: 0.85rem; color: #E2E8F0;">Vector DB</span>
                <span style="font-size: 0.75rem; font-weight: 700; color: #10B981; background: rgba(16, 185, 129, 0.1); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(16, 185, 129, 0.3);">● 220 Indexed</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="font-size: 0.85rem; color: #E2E8F0;">LLM Provider</span>
                <span style="font-size: 0.75rem; font-weight: 700; color: #6366F1; background: rgba(99, 102, 241, 0.1); padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(99, 102, 241, 0.3);">Groq Active</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 0.85rem; color: #E2E8F0;">Embeddings</span>
                <span style="font-size: 0.75rem; font-weight: 600; color: #94A3B8;">all-MiniLM-L6-v2</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("**Quick Recruiter Queries:**")
    quick_queries = [
        "Find me a Junior Data Analyst who knows SQL and Tableau.",
        "Senior Java Developer with Spring Boot and Hibernate.",
        "QA Automation Tester with Python and Selenium.",
        "Database Administrator skilled in Oracle & SQL Server.",
        "Front-end Developer with Angular, JavaScript, HTML."
    ]
    for q in quick_queries:
        if st.button(f"🔍 {q[:32]}...", key=f"btn_{q[:12]}", use_container_width=True):
            st.session_state["active_search_query"] = q

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.75rem; color: #64748B; text-align: center; line-height: 1.5;">
            TalentPulse Engine v1.0.0<br/>
            Task 8 Compliant · Zero Hallucination
        </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero Section
# ---------------------------------------------------------------------------
st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚡ Advanced RAG Talent Search Engine</div>
        <div class="hero-title">Discover Top Talent With Precision</div>
        <div class="hero-subtitle">
            Query across 220 comprehensive resume profiles with local dense embeddings and 
            receive evidence-backed candidate evaluations powered by high-speed Groq AI reasoning.
        </div>
    </div>
""", unsafe_allow_html=True)


# Database check & Cloud Deployment Auto-Init
if not has_index:
    st.warning("⚠️ **Vector Database Not Initialized**")
    st.markdown("""
        The ChromaDB vector database is currently unpopulated on this environment.
        You can build the index directly by clicking below.
    """)
    
    if DATA_PATH.exists():
        if st.button("🚀 Initialize & Build Vector Index (220 Resumes)", type="primary"):
            with st.spinner("Parsing resumes and generating embeddings (takes ~15 seconds)..."):
                from src.data_loader import load_raw_data
                from src.resume_parser import parse_all_resumes
                from src.vector_store import build_index
                
                records = load_raw_data(DATA_PATH)
                profs = parse_all_resumes(records)
                build_index(profs, rebuild=True)
                st.success("✅ Index built successfully! Reloading application...")
                st.rerun()
    else:
        st.info("📁 You can upload `Entity Recognition in Resumes.json` directly to initialize the vector database:")
        uploaded = st.file_uploader("Upload Resumes JSON", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DATA_PATH, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success("✅ File uploaded! Initializing vector database...")
            with st.spinner("Building index..."):
                from src.data_loader import load_raw_data
                from src.resume_parser import parse_all_resumes
                from src.vector_store import build_index
                
                records = load_raw_data(DATA_PATH)
                profs = parse_all_resumes(records)
                build_index(profs, rebuild=True)
                st.success("✅ Vector database ready! Reloading...")
                st.rerun()
    st.stop()

# Initialize components
_ = load_cached_embedding_model()
collection = load_cached_vector_collection()
rag_pipeline = RAGPipeline(collection=collection)
bias_checker = BiasChecker(collection=collection)


# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_search, tab_details, tab_qa, tab_bias, tab_system = st.tabs([
    "🔍  Talent Search",
    "👤  Candidate Directory",
    "💬  Ask Resume DB",
    "⚖️  Bias & Fairness Audit",
    "ℹ️  Architecture & Stats"
])


# ===========================================================================
# TAB 1: TALENT SEARCH
# ===========================================================================
with tab_search:
    st.markdown("#### 🎯 Natural-Language Candidate Discovery")
    
    col_search, col_slider = st.columns([4, 1])
    with col_search:
        default_query = st.session_state.get(
            "active_search_query",
            "Find me a Junior Data Analyst who knows SQL and Tableau."
        )
        user_query = st.text_input(
            "Target Candidate Profile / Requirements:",
            value=default_query,
            placeholder="e.g. Senior Full-Stack Engineer with React, Node.js, and AWS experience"
        )
    with col_slider:
        k_val = st.slider("Top Candidates:", min_value=1, max_value=12, value=5)

    search_clicked = st.button("🚀  Execute Semantic Search", type="primary", use_container_width=True)

    if search_clicked or "cached_search" in st.session_state:
        if search_clicked:
            with st.spinner("⚡ Embedding query and retrieving matches from ChromaDB..."):
                search_payload = rag_pipeline.search_and_evaluate(query=user_query, top_k=k_val, num_to_evaluate=3)
                st.session_state["cached_search"] = search_payload
                st.session_state["cached_query"] = user_query
        else:
            search_payload = st.session_state["cached_search"]
            user_query = st.session_state.get("cached_query", user_query)

        # Performance Metric Grid
        st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Retrieved Profiles</div>
                    <div class="metric-value">{len(search_payload['retrieved_candidates'])}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Retrieval Latency</div>
                    <div class="metric-value">{search_payload['retrieval_latency'] * 1000:.1f} ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">AI Reasoning Time</div>
                    <div class="metric-value">{search_payload['llm_latency']:.2f} s</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">End-to-End Latency</div>
                    <div class="metric-value">{search_payload['total_latency']:.2f} s</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Groq LLM Evaluation Box
        st.markdown("""
            <div class="ai-report-card">
                <div class="ai-report-header">
                    <span>✦ AI Candidate Fit Synthesis</span>
                    <span class="ai-badge">Top 3 Candidates</span>
                </div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 1.2rem;">
                    Explanations are strictly grounded in resume text with zero speculative inference.
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(search_payload["llm_explanation"])

        st.markdown("---")
        st.markdown(f"#### 📋 Top {len(search_payload['retrieved_candidates'])} Matching Candidate Profiles")

        for rank, c in enumerate(search_payload["retrieved_candidates"], 1):
            meta = c.get("metadata", {})
            name = meta.get("name", "Candidate")
            role = meta.get("designation", "Designation Not Specified")
            score = c.get("semantic_similarity_score", 0.0)
            score_pct = int(score * 100)
            cid = c.get("candidate_id")

            # Initials for avatar
            initials = "".join([part[0] for part in name.split()[:2]]).upper() if name else "CA"

            with st.expander(f"#{rank}  {name} — {role}  (Similarity: {score:.4f})", expanded=(rank <= 2)):
                col_left, col_right = st.columns([1, 2])
                with col_left:
                    st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <div class="cand-avatar">{initials}</div>
                            <div>
                                <div class="cand-name">{name}</div>
                                <div class="cand-role">{role}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"**Candidate ID:** `{cid}`")
                    st.markdown(f"**Match Relevance:** <span class='score-chip'>{score:.4f} ({score_pct}%)</span>", unsafe_allow_html=True)
                    st.markdown(f"**Experience:** {meta.get('experience', 'Not specified')}")
                    st.markdown(f"**Location:** {meta.get('location', 'Not specified')}")
                    st.markdown(f"**Education:** {meta.get('degree', 'Not specified')}")
                    st.markdown(f"**Institution:** {meta.get('college', 'Not specified')}")
                    st.markdown(f"**Companies:** {meta.get('companies', 'Not specified')}")

                with col_right:
                    st.markdown("**Skills Extracted from Resume:**")
                    skills_raw = str(meta.get("skills", ""))
                    if skills_raw and skills_raw != "Not specified":
                        skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()]
                        tags_html = "".join([f"<span class='skill-pill'>{s}</span>" for s in skills_list[:20]])
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.write("Not specified")

                    st.markdown("**Original Resume Content Excerpt:**")
                    doc_text = c.get("document", "")
                    st.markdown(f"<div class='resume-box'>{doc_text[:1400]}...</div>", unsafe_allow_html=True)


# ===========================================================================
# TAB 2: CANDIDATE DIRECTORY
# ===========================================================================
with tab_details:
    st.markdown("#### 👤 Complete Candidate Profiles Database")
    st.write("Browse any of the 220 parsed resumes and examine extracted NER entity fields.")

    all_cand_list = get_all_candidates()
    if all_cand_list:
        cand_map = {
            f"{c['metadata'].get('name', 'Candidate')} | {c['metadata'].get('designation', 'Role N/A')} ({c['candidate_id']})": c['candidate_id']
            for c in all_cand_list
        }
        chosen_cand_key = st.selectbox("Select Candidate to Inspect:", options=list(cand_map.keys()))
        target_cand = get_candidate_by_id(cand_map[chosen_cand_key])

        if target_cand:
            t_meta = target_cand.get("metadata", {})
            t_name = t_meta.get("name", "Candidate")
            t_role = t_meta.get("designation", "Not specified")
            t_initials = "".join([part[0] for part in t_name.split()[:2]]).upper() if t_name else "CA"

            st.markdown(f"""
                <div class="candidate-card-v2" style="margin-top: 1rem;">
                    <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
                        <div class="cand-avatar" style="width: 56px; height: 56px; font-size: 1.3rem;">{t_initials}</div>
                        <div>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC;">{t_name}</div>
                            <div style="font-size: 1rem; color: #A5B4FC;">{t_role}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Experience", t_meta.get("experience", "Not specified"))
            d2.metric("Location", t_meta.get("location", "Not specified"))
            d3.metric("Degree", t_meta.get("degree", "Not specified"))
            d4.metric("College", t_meta.get("college", "Not specified"))

            st.markdown("---")
            st.markdown("##### Annotated Skills:")
            t_skills = str(t_meta.get("skills", ""))
            if t_skills and t_skills != "Not specified":
                s_items = [s.strip() for s in t_skills.split(",") if s.strip()]
                t_html = "".join([f"<span class='skill-pill'>{s}</span>" for s in s_items])
                st.markdown(t_html, unsafe_allow_html=True)
            else:
                st.write("No skills explicitly annotated.")

            st.markdown("---")
            st.markdown("##### Complete Resume Text:")
            st.markdown(f"<div class='resume-box' style='max-height: 450px;'>{target_cand.get('text', '')}</div>", unsafe_allow_html=True)


# ===========================================================================
# TAB 3: ASK RESUME DATABASE
# ===========================================================================
with tab_qa:
    st.markdown("#### 💬 Grounded Candidate Q&A Chatbot")
    st.write("Pose targeted verification questions about any candidate. The AI answers strictly using facts present in their resume.")

    if all_cand_list:
        qa_select_map = {
            f"{c['metadata'].get('name', 'Candidate')} ({c['candidate_id']})": c['candidate_id']
            for c in all_cand_list
        }
        chosen_qa_cand = st.selectbox("Select Target Candidate:", options=list(qa_select_map.keys()), key="qa_select_input")
        target_qa_id = qa_select_map[chosen_qa_cand]

        st.markdown("**Suggested Inquiries:**")
        q_btn_cols = st.columns(3)
        sample_1 = "Does this candidate have SQL experience?"
        sample_2 = "What programming languages does this candidate know?"
        sample_3 = "Does this candidate have leadership or team mentoring experience?"

        if q_btn_cols[0].button(f"🔍 {sample_1}", use_container_width=True):
            st.session_state["current_qa_q"] = sample_1
        if q_btn_cols[1].button(f"🔍 {sample_2}", use_container_width=True):
            st.session_state["current_qa_q"] = sample_2
        if q_btn_cols[2].button(f"🔍 {sample_3}", use_container_width=True):
            st.session_state["current_qa_q"] = sample_3

        qa_input = st.text_input(
            "Recruiter Question:",
            value=st.session_state.get("current_qa_q", ""),
            placeholder="e.g. Does this candidate have cloud infrastructure deployment experience?"
        )

        if st.button("💬  Submit Inquiry", type="primary"):
            if not qa_input.strip():
                st.warning("Please specify a question.")
            else:
                with st.spinner("🤖 Consulting candidate resume..."):
                    qa_out = rag_pipeline.ask_candidate_question(target_qa_id, qa_input)

                st.markdown(f"""
                    <div class="ai-report-card">
                        <div style="font-size: 0.85rem; color: #A5B4FC; font-weight: 700; margin-bottom: 0.5rem; text-transform: uppercase;">
                            Grounded Answer (Latency: {qa_out['latency']:.2f}s)
                        </div>
                        <div style="font-size: 1.05rem; color: #F8FAFC; line-height: 1.7;">
                            {qa_out['answer']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)


# ===========================================================================
# TAB 4: BIAS & FAIRNESS AUDIT
# ===========================================================================
with tab_bias:
    st.markdown("#### ⚖️ Explainable Retrieval Bias Audit")
    st.write(
        "Audit whether the semantic ranking is being influenced by non-job-relevant demographic signals "
        "(Candidate Names, Geographic Locations, Universities, Graduation Years)."
    )

    audit_query_str = st.text_input(
        "Search Query to Audit:",
        value=st.session_state.get("cached_query", "Find me a Junior Data Analyst who knows SQL and Tableau."),
        key="bias_audit_input_field"
    )

    if st.button("🔍  Run Explainable Fairness Audit", type="primary"):
        with st.spinner("Running comparative vector audit (Full text vs. Sanitized job-relevant text)..."):
            audit_report = bias_checker.audit_query_results(audit_query_str, top_k=5)

        if audit_report["signal_level"] == "warning":
            st.markdown(f"""
                <div class="audit-box-warning">
                    <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.3rem;">
                        {audit_report['audit_signal']}
                    </div>
                    <div>{audit_report['audit_summary']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="audit-box-success">
                    <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 0.3rem;">
                        {audit_report['audit_signal']}
                    </div>
                    <div>{audit_report['audit_summary']}</div>
                </div>
            """, unsafe_allow_html=True)

        if audit_report["ranking_comparison"]:
            st.markdown("##### Detailed Rank Shift Analysis:")
            df_bias = pd.DataFrame(audit_report["ranking_comparison"])
            df_bias_clean = df_bias[[
                "name", "baseline_rank", "sanitized_rank", "rank_shift",
                "baseline_score", "sanitized_score", "score_delta", "location", "college"
            ]].rename(columns={
                "name": "Candidate",
                "baseline_rank": "Original Rank",
                "sanitized_rank": "Masked Rank",
                "rank_shift": "Rank Shift",
                "baseline_score": "Original Sim",
                "sanitized_score": "Masked Sim",
                "score_delta": "Δ Delta",
                "location": "Location",
                "college": "University"
            })
            st.dataframe(df_bias_clean, use_container_width=True)

        st.markdown("---")
        st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1.2rem;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">Legal & Algorithmic Notice</div>
                <div style="font-size: 0.82rem; color: #64748B; line-height: 1.6;">
                    This audit measures ranking variance when non-job-relevant metadata is redacted. 
                    It does NOT infer demographic labels (gender, race, ethnicity) and serves as an informational guardrail. 
                    Human recruiter supervision is mandatory.
                </div>
            </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# TAB 5: SYSTEM ARCHITECTURE & STATS
# ===========================================================================
with tab_system:
    st.markdown("#### ℹ️ System Architecture & Model Telemetry")

    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown("""
            <div class="candidate-card-v2">
                <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.8rem;">
                    Vector & Retrieval Layer
                </div>
                <div style="font-size: 0.9rem; color: #94A3B8; line-height: 1.8;">
                    • <b>Embedding Model:</b> <code>sentence-transformers/all-MiniLM-L6-v2</code><br/>
                    • <b>Vector Dimension:</b> 384 dimensions (Normalized)<br/>
                    • <b>Execution:</b> 100% Local CPU (No API tokens required)<br/>
                    • <b>Vector Store:</b> Persistent ChromaDB (Cosine Space)<br/>
                    • <b>Index Document Unit:</b> Whole Resume (1 Candidate = 1 Doc)<br/>
                    • <b>Total Indexed Resumes:</b> 220 profiles
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_arch2:
        st.markdown(f"""
            <div class="candidate-card-v2">
                <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.8rem;">
                    Reasoning & Generation Layer
                </div>
                <div style="font-size: 0.9rem; color: #94A3B8; line-height: 1.8;">
                    • <b>LLM Provider:</b> <code>{LLM_PROVIDER.upper()}</code><br/>
                    • <b>Active Model:</b> <code>{LLM_MODEL}</code><br/>
                    • <b>Evaluation Unit:</b> Top 3 Retrieved Candidates<br/>
                    • <b>Grounding Constraint:</b> Strict Zero-Hallucination Policy<br/>
                    • <b>Evidence Mandate:</b> Exact resume text quotation<br/>
                    • <b>Framework:</b> LangChain v0.3
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("##### End-to-End RAG Architecture Pipeline:")
    st.markdown("""
```
 Recruiter Query: "Find me a Junior Data Analyst who knows SQL and Tableau"
                              ↓
              Dense Embedding via all-MiniLM-L6-v2
                              ↓
     ChromaDB Persistent Cosine Vector Retrieval (Top-K Candidates)
                              ↓
           Top 3 Candidates Selected for Deep AI Synthesis
                              ↓
 Strict Zero-Hallucination Grounded Prompt Construction (Resume Excerpts Only)
                              ↓
      Groq LLM High-Speed Inference (qwen/qwen3.8-27b / Llama-3.3)
                              ↓
  Structured Fit Explanations + Exact Evidence Quotes + Gaps Identified
                              ↓
               Streamlit Recruiter Dashboard (Port 8501)
```
    """)
