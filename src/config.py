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

# ---------------------------------------------------------------------------
# LLM Provider Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Provider-specific models
LLM_MODELS = {
    "groq": os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "google": os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
}

LLM_MODEL = LLM_MODELS.get(LLM_PROVIDER, "qwen/qwen3.8-27b")

# API Keys (loaded from environment)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Temperature
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ---------------------------------------------------------------------------
# Miscellaneous
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Sensitive / Non-Job-Relevant Fields (for Bias Check)
# ---------------------------------------------------------------------------
SENSITIVE_FIELDS = ["name", "location", "graduation_year", "college", "email"]
JOB_RELEVANT_FIELDS = ["skills", "designation", "experience", "degree", "companies"]
