"""Simple web search tool."""

from __future__ import annotations

import re
from html import unescape

import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from pydantic import PrivateAttr


class SearchInput(BaseModel):
    """Expected search arguments."""

    query: str = Field(..., description="Search query")
    num_results: int = Field(3, description="Maximum number of search results")


class SearchWebTool(BaseTool):
    """Performs a lightweight HTML search request."""

    name: str = "search_web"
    description: str = (
        "Ищет информацию в интернете по текстовому запросу и возвращает краткий список результатов."
    )
    args_schema: type[BaseModel] = SearchInput
    _base_url: str = PrivateAttr()

    def __init__(self, base_url: str = "https://duckduckgo.com/html/", **kwargs) -> None:
        super().__init__(**kwargs)
        self._base_url = base_url

    def _parse_results(self, html: str, num_results: int) -> list[str]:
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        matches = pattern.finditer(html)

        parsed = []
        for match in matches:
            title = re.sub(r"<.*?>", "", match.group("title"))
            href = unescape(match.group("href"))
            parsed.append(f"{unescape(title)} - {href}")
            if len(parsed) >= num_results:
                break

        return parsed

    def _run(self, query: str, num_results: int = 3) -> str:
        try:
            response = requests.get(
                self._base_url,
                params={"q": query},
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
        except requests.RequestException as error:
            return f"Поиск временно недоступен: {error}"

        results = self._parse_results(response.text, num_results=num_results)
        if not results:
            return "По запросу не найдено результатов."
        return "\n".join(results)

    async def _arun(self, query: str, num_results: int = 3) -> str:
        return self._run(query=query, num_results=num_results)
