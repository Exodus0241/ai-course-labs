"""Date utility tool for month calculations."""

from __future__ import annotations

import re
from datetime import datetime

from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class MonthsSinceInput(BaseModel):
    """Expected arguments for month difference calculation."""

    date_text: str = Field(
        ...,
        description="Release date in formats like YYYY-MM-DD, YYYY-MM, March 2025 or 31 March 2025",
    )


class MonthsSinceDateTool(BaseTool):
    """Calculates how many whole months passed since a given release date."""

    name: str = "months_since_date"
    description: str = (
        "Считает, сколько месяцев прошло с указанной даты до сегодняшнего дня. "
        "Используй после search_web, когда уже найдена дата релиза или обновления. "
        "Передавай только дату, например: 2025-08-15, 2025-08 или March 2025."
    )
    args_schema: type[BaseModel] = MonthsSinceInput

    def _parse_date(self, date_text: str) -> datetime:
        """Parses a date from several user-friendly formats."""
        cleaned = date_text.strip()
        formats = [
            "%Y-%m-%d",
            "%Y-%m",
            "%d.%m.%Y",
            "%d-%m-%Y",
            "%d %B %Y",
            "%B %Y",
            "%b %Y",
        ]

        normalized = re.sub(r"\s+", " ", cleaned)
        for fmt in formats:
            try:
                parsed = datetime.strptime(normalized, fmt)
                if fmt in {"%Y-%m", "%B %Y", "%b %Y"}:
                    return parsed.replace(day=1)
                return parsed
            except ValueError:
                continue

        raise ValueError(
            "Не удалось распознать дату. Используй формат YYYY-MM-DD, YYYY-MM или, например, March 2025."
        )

    def _run(self, date_text: str) -> str:
        release_date = self._parse_date(date_text)
        today = datetime.now()
        months = (today.year - release_date.year) * 12 + (today.month - release_date.month)
        if today.day < release_date.day:
            months -= 1
        return f"{max(months, 0)}"

    async def _arun(self, date_text: str) -> str:
        return self._run(date_text=date_text)
