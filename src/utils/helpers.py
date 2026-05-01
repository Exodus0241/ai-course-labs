"""Small helpers for demos, tests and report generation."""

from __future__ import annotations

from typing import Dict, Iterable, List


def format_source_table_rows(results: Iterable[Dict]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for index, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})
        rows.append(
            {
                "rank": str(index),
                "source": metadata.get("file_name", metadata.get("source", "unknown")),
                "chunk_id": str(metadata.get("chunk_id", "")),
                "score": str(item.get("similarity_score", "")),
            }
        )
    return rows


def markdown_checklist(items: Iterable[tuple[str, bool]]) -> str:
    return "\n".join(f"- [{'x' if done else ' '}] {title}" for title, done in items)
