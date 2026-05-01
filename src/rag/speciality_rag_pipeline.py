"""Domain-specific RAG pipeline for milk packaging defect control."""

from __future__ import annotations

from typing import Any, Optional

from src.rag.rag_pipeline import RAGPipeline
from src.rag.vector_store import VectorStoreManager


class DairyPackagingRAGPipeline(RAGPipeline):
    """Specialized prompt for diploma subject area."""

    def __init__(
        self,
        vectorstore: VectorStoreManager,
        llm: Optional[Any] = None,
        top_k: int = 4,
    ) -> None:
        system_prompt = (
            "Ты экспертный помощник по информационной системе контроля дефектов закупорки "
            "молочной тары на производстве. Анализируй только переданные документы. "
            "Если вопрос связан с причинами дефекта, укажи возможные факторы процесса: "
            "крышка, укупорочный узел, конвейер, датчики, операторский контроль, журнал инцидентов. "
            "Если вопрос связан с ИС, выделяй сущности системы: дефект, партия, линия, оборудование, событие, уведомление. "
            "Не выдумывай факты вне контекста."
        )
        super().__init__(
            vectorstore=vectorstore,
            llm=llm,
            top_k=top_k,
            system_prompt=system_prompt,
        )


SpecialityRAGPipeline = DairyPackagingRAGPipeline
