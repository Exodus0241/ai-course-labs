"""Utility helpers for formatting and report data."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def format_facts(facts: Dict[str, Any]) -> str:
    """Format facts as key-value lines."""
    return "\n".join(f"- {key}: {value}" for key, value in facts.items())


def format_rule_names(rule_dicts: Iterable[Dict[str, Any]]) -> str:
    """Join rule names for a compact report."""
    names = [rule["name"] for rule in rule_dicts]
    return ", ".join(names) if names else "нет"
