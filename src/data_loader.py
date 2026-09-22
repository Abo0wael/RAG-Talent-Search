"""
RAG Talent Search Engine - Data Loader

Loads the DataTurks-format resume NER dataset from a JSON-lines file.
Each line in the file is a separate JSON object with:
  - "content": the raw resume text
  - "annotation": list of entity annotations with label, start, end, text
  - "extras": (usually null)
"""

import json
from pathlib import Path

from src.config import DATA_PATH


def load_raw_data(data_path: str | Path | None = None) -> list[dict]:
    """
    Load raw resume records from the DataTurks JSON-lines file.

    Args:
        data_path: Path to the dataset file. Defaults to config.DATA_PATH.

    Returns:
        List of raw record dicts, each with 'content' and 'annotation' keys.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file is empty or contains no valid records.
    """
    path = Path(data_path) if data_path else DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"Dataset file not found:\n"
            f"  {path}\n\n"
            f"To fix this:\n"
            f"1. Go to: https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner\n"
            f"2. Download 'Entity Recognition in Resumes.json'\n"
            f"3. Place it in: {path.parent}/\n"
            f"{'='*60}"
        )

    records = []
    errors = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Validate minimal structure
                if "content" in record and record["content"]:
                    records.append(record)
                else:
                    errors.append(f"Line {line_num}: missing or empty 'content' field")
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON - {e}")

    if errors:
        print(f"[DataLoader] Warning: {len(errors)} lines had issues:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    if not records:
        raise ValueError(
            f"No valid resume records found in {path}. "
            f"The file may be corrupted or in an unexpected format."
        )

    print(f"[DataLoader] Successfully loaded {len(records)} resumes from {path.name}")
    return records


def get_dataset_stats(records: list[dict]) -> dict:
    """
    Compute basic statistics about the loaded dataset.

    Args:
        records: List of raw records from load_raw_data().

    Returns:
        Dict with statistics like total count, avg text length, label counts.
    """
    if not records:
        return {"total_resumes": 0}

    text_lengths = [len(r.get("content", "")) for r in records]

    # Count annotation labels
    label_counts: dict[str, int] = {}
    for record in records:
        annotations = record.get("annotation") or []
        for ann in annotations:
            labels = ann.get("label", [])
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "total_resumes": len(records),
        "avg_text_length": sum(text_lengths) / len(text_lengths),
        "min_text_length": min(text_lengths),
        "max_text_length": max(text_lengths),
        "label_counts": dict(sorted(label_counts.items(), key=lambda x: -x[1])),
    }
