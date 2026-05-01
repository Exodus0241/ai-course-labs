"""Core retrieval-augmented generation pipeline."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.rag.vector_store import VectorStoreManager

LOGGER = logging.getLogger(__name__)


class RAGPipeline:
    """Searches documents and optionally calls an LLM with retrieved context."""

    def __init__(
        self,
        vectorstore: VectorStoreManager,
        llm: Optional[Any] = None,
        top_k: int = 4,
        system_prompt: Optional[str] = None,
    ) -> None:
        if llm is None:
            raise ValueError("Для RAGPipeline необходимо передать инициализированный LLM.")
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k
        self.system_prompt = system_prompt or (
            "Ты аналитический ассистент по производственной документации. "
            "Отвечай только на основе переданного контекста. "
            "Если данных недостаточно, прямо скажи, что в документах нет точного ответа. "
            "В ответе сначала дай краткий вывод, затем при необходимости перечисли подтверждающие факты."
        )

    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        started_at = time.perf_counter()
        search_results = self.vectorstore.search_with_scores(query=question, k=self.top_k)

        if not search_results:
            return {
                "success": False,
                "question": question,
                "answer": "Релевантные документы не найдены.",
                "sources": [],
                "sources_count": 0,
                "execution_time": round(time.perf_counter() - started_at, 4),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }

        context = self._format_context(search_results)
        answer = self._generate_answer(question=question, context=context)

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": search_results if include_sources else [],
            "sources_count": len(search_results),
            "execution_time": round(time.perf_counter() - started_at, 4),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def query_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        return [self.query(question) for question in questions]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "llm_enabled": self.llm is not None,
            "vectorstore": self.vectorstore.get_statistics(),
        }

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        formatted_chunks = []
        for item in documents:
            source = item["metadata"].get("file_name", item["metadata"].get("source", "unknown"))
            chunk_id = item["metadata"].get("chunk_id", "?")
            formatted_chunks.append(
                f"[Источник: {source}, chunk {chunk_id}, score {item['similarity_score']}]\n"
                f"{item['content']}"
            )
        return "\n\n".join(formatted_chunks)

    def _generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Контекст:\n{context}\n\n"
            f"Вопрос: {question}\n"
            "Ответ:"
        )

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return str(response.content).strip()
            return str(response).strip()
        except Exception as exc:
            LOGGER.error("Ошибка вызова LLM: %s", exc)
            raise RuntimeError(f"Ошибка генерации ответа через LLM: {exc}") from exc
