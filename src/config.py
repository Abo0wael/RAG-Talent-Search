"""
RAG Talent Search Engine - Centralized Configuration

All configurable parameters are defined here.
Secrets are loaded from .env via python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file (project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = _PROJECT_ROOT
DATA_PATH = _PROJECT_ROOT / "data" / "Entity Recognition in Resumes.json"
VECTOR_DB_PATH = str(_PROJECT_ROOT / "vectorstore")
OUTPUTS_DIR = _PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EVALUATION_DIR = OUTPUTS_DIR / "evaluation"

# ---------------------------------------------------------------------------
# Embedding Model (runs locally, no API key needed)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Retrieval Settings
# ---------------------------------------------------------------------------
TOP_K = 5                     # Default number of candidates to retrieve
COLLECTION_NAME = "resumes"   # ChromaDB collection name

def _get_secret(key: str, default: str | None = None) -> str | None:
    """Get secret from os.environ, falling back to st.secrets for Streamlit Cloud."""
    val = os.getenv(key)
    if val and val != f"your_{key.lower()}_here":
        return val
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            s_val = st.secrets[key]
            if s_val:
                os.environ[key] = s_val
                return s_val
    except Exception:
        pass
    return default


# ---------------------------------------------------------------------------
# LLM Provider Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = _get_secret("LLM_PROVIDER", "groq").lower()

# Provider-specific models
LLM_MODELS = {
    "groq": _get_secret("GROQ_MODEL", "qwen/qwen3.8-27b"),
    "openai": _get_secret("OPENAI_MODEL", "gpt-4o-mini"),
    "google": _get_secret("GOOGLE_MODEL", "gemini-1.5-flash"),
}

LLM_MODEL = LLM_MODELS.get(LLM_PROVIDER, "qwen/qwen3.8-27b")

# API Keys (loaded from environment or st.secrets on cloud)
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
GOOGLE_API_KEY = _get_secret("GOOGLE_API_KEY")

# LLM Temperature
LLM_TEMPERATURE = float(_get_secret("LLM_TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Sensitive / Non-Job-Relevant Fields (for Bias Check)
# ---------------------------------------------------------------------------
SENSITIVE_FIELDS = ["name", "location", "graduation_year", "college", "email"]
JOB_RELEVANT_FIELDS = ["skills", "designation", "experience", "degree", "companies"]
