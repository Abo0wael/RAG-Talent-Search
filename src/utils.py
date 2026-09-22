"""
RAG Talent Search Engine - Utility Functions

Text cleaning, timing helpers, and formatting utilities.
"""

import re
import time
import functools
from typing import Any


def clean_text(text: str) -> str:
    """
    Clean resume text for embedding and display.

    - Remove URLs
    - Collapse excessive whitespace / newlines
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove email-style Indeed links
    text = re.sub(r'indeed\.com/r/[\w\-/]+', '', text)

    # Collapse multiple newlines into at most two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse multiple spaces into one
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Strip each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length characters with ellipsis."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length].rsplit(' ', 1)[0] + "..."


def timer(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed
    return wrapper


class Timer:
    """Context manager for timing blocks of code."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start


def format_score(score: float) -> str:
    """Format a similarity score for display."""
    return f"{score:.4f}"


def format_skills_list(skills: list[str] | str | None) -> str:
    """Format skills as a readable comma-separated string."""
    if not skills:
        return "Not specified"
    if isinstance(skills, str):
        return skills
    return ", ".join(skills)


def format_companies_list(companies: list[str] | str | None) -> str:
    """Format companies as a readable comma-separated string."""
    if not companies:
        return "Not specified"
    if isinstance(companies, str):
        return companies
    return ", ".join(companies)


def safe_json_value(value: Any) -> Any:
    """Convert a value to a JSON-safe representation for ChromaDB metadata."""
    if value is None:
        return "Not specified"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "Not specified"
    return str(value)
