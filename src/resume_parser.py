"""
RAG Talent Search Engine - Resume Parser

Parses DataTurks-annotated resume records into a normalized CandidateProfile.

Each record in the dataset has:
  - "content": full resume text
  - "annotation": list of dicts with {"label": [...], "points": [{"start", "end", "text"}]}

The parser extracts entities by label and builds a structured profile.
Missing information is stored as None (not fabricated).
"""

import re
import hashlib
from typing import Any

from src.utils import clean_text


# ── Internal helpers ────────────────────────────────────────────────────────

def _extract_annotations_by_label(annotations: list[dict] | None) -> dict[str, list[str]]:
    """
    Group annotation text values by their label.

    Returns:
        Dict mapping label → list of text strings.
        Example: {"Skills": ["Python", "SQL"], "Name": ["John Doe"]}
    """
    if not annotations:
        return {}

    grouped: dict[str, list[str]] = {}
    for ann in annotations:
        labels = ann.get("label", [])
        points = ann.get("points", [])
        for label in labels:
            for pt in points:
                text = pt.get("text", "").strip()
                if text:
                    grouped.setdefault(label, []).append(text)
    return grouped


def _first_or_none(values: list[str] | None) -> str | None:
    """Return the first non-empty value or None."""
    if not values:
        return None
    for v in values:
        v = v.strip()
        if v:
            return v
    return None


def _unique_list(values: list[str] | None) -> list[str]:
    """De-duplicate and clean a list of strings, preserving order."""
    if not values:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for v in values:
        v = v.strip().rstrip('\n')
        v_lower = v.lower()
        if v and v_lower not in seen:
            seen.add(v_lower)
            result.append(v)
    return result


def _parse_skills_text(raw_skills: list[str]) -> list[str]:
    """
    Parse raw skills annotations into individual skill items.

    The dataset often has multi-line skill blocks like:
      "C (Less than 1 year), Java (Less than 1 year)"
    or bullet-pointed lists.
    """
    all_skills: list[str] = []

    for block in raw_skills:
        # Remove experience qualifiers like "(Less than 1 year)"
        block = re.sub(r'\((?:Less than|More than)?\s*\d+\s*years?\)', '', block, flags=re.IGNORECASE)
        block = re.sub(r'\(\d+\s*years?\)', '', block, flags=re.IGNORECASE)

        # Split on common delimiters
        parts = re.split(r'[,\n•●◆☑➢❖▪▸]', block)

        for part in parts:
            # Remove bullet markers and clean
            skill = re.sub(r'^[\s\-\*\•\●\◆]+', '', part).strip()
            # Remove header-like prefixes
            skill = re.sub(r'^(Programming [Ll]anguages?|Technical [Ss]kills?|'
                          r'Web [Dd]evelopment|[Dd]atabase|'
                          r'Operating [Ss]ystems?|Languages?|'
                          r'[Tt]echnologies|[Ff]rameworks?|Tools?|IDE|'
                          r'[Ss]oftware)\s*[:]\s*', '', skill).strip()
            if skill and len(skill) > 1 and len(skill) < 80:
                all_skills.append(skill)

    return _unique_list(all_skills)


def _generate_candidate_id(index: int, name: str | None, text: str) -> str:
    """Generate a stable candidate ID from index and content hash."""
    hash_input = f"{index}:{text[:200]}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    if name:
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name.strip())[:20]
        return f"candidate_{index:03d}_{safe_name}_{short_hash}"
    return f"candidate_{index:03d}_{short_hash}"


# ── Public API ──────────────────────────────────────────────────────────────

def parse_resume(record: dict, index: int = 0) -> dict[str, Any]:
    """
    Parse a single DataTurks resume record into a normalized CandidateProfile.

    Args:
        record: Raw record dict with 'content' and 'annotation' keys.
        index: The resume's position in the dataset (for ID generation).

    Returns:
        Normalized profile dict:
        {
            "candidate_id": str,
            "text": str,              # cleaned full resume text
            "name": str | None,
            "skills": list[str],
            "designation": str | None,
            "experience": str | None,
            "degree": str | None,
            "college": str | None,
            "graduation_year": str | None,
            "companies": list[str],
            "location": str | None,
            "email": str | None,
        }
    """
    raw_text = record.get("content", "")
    annotations = record.get("annotation") or []

    # Group annotations by label
    by_label = _extract_annotations_by_label(annotations)

    # Extract fields
    name = _first_or_none(by_label.get("Name"))
    designation = _first_or_none(by_label.get("Designation"))
    experience = _first_or_none(by_label.get("Years of Experience"))
    degree = _first_or_none(by_label.get("Degree"))
    college = _first_or_none(by_label.get("College Name"))
    graduation_year = _first_or_none(by_label.get("Graduation Year"))
    location = _first_or_none(by_label.get("Location"))
    email = _first_or_none(by_label.get("Email Address"))

    # Parse skills (complex multi-block extraction)
    skills = _parse_skills_text(by_label.get("Skills", []))

    # Parse companies
    companies = _unique_list(by_label.get("Companies worked at", []))

    # Clean the full resume text
    cleaned_text = clean_text(raw_text)

    # Generate candidate ID
    candidate_id = _generate_candidate_id(index, name, raw_text)

    return {
        "candidate_id": candidate_id,
        "text": cleaned_text,
        "name": name,
        "skills": skills,
        "designation": designation,
        "experience": experience,
        "degree": degree,
        "college": college,
        "graduation_year": graduation_year,
        "companies": companies,
        "location": location,
        "email": email,
    }


def parse_all_resumes(records: list[dict]) -> list[dict[str, Any]]:
    """
    Parse all raw records into normalized CandidateProfiles.

    Args:
        records: List of raw records from data_loader.load_raw_data().

    Returns:
        List of normalized profile dicts.
    """
    profiles = []
    for i, record in enumerate(records):
        try:
            profile = parse_resume(record, index=i)
            profiles.append(profile)
        except Exception as e:
            print(f"[ResumeParser] Warning: failed to parse resume {i}: {e}")

    print(f"[ResumeParser] Successfully parsed {len(profiles)} / {len(records)} resumes")
    return profiles
