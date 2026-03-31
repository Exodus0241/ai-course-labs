"""Input validation rules for the agent."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Represents the result of user input validation."""

    is_valid: bool
    cleaned_text: str
    reason: str = ""


class InputValidator:
    """Applies simple guardrails before the request reaches the agent."""

    def __init__(self, max_length: int = 1_000) -> None:
        self.max_length = max_length
        self.blocked_patterns = [
            r"ignore\s+previous\s+instructions",
            r"system\s+prompt",
            r"rm\s+-rf",
            r"drop\s+table",
            r"<script.*?>.*?</script>",
        ]

    def validate(self, text: str) -> ValidationResult:
        """Checks size, emptiness and obvious malicious prompt patterns."""
        cleaned = " ".join(text.split())

        if not cleaned:
            return ValidationResult(False, "", "Запрос пустой.")

        if len(cleaned) > self.max_length:
            return ValidationResult(
                False,
                cleaned[: self.max_length],
                f"Запрос слишком длинный. Максимум: {self.max_length} символов.",
            )

        for pattern in self.blocked_patterns:
            if re.search(pattern, cleaned, flags=re.IGNORECASE):
                return ValidationResult(
                    False,
                    cleaned,
                    "Запрос отклонён guardrails из-за потенциально опасного содержимого.",
                )

        return ValidationResult(True, cleaned)
