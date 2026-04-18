"""Research agent responsible for collecting source facts and context."""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentExecutionResult, BaseSpecializedAgent
from src.memory.semantic_memory import SemanticMemory
from src.tools.search_tool import SearchWebTool


class ResearcherAgent(BaseSpecializedAgent):
    """Collects external and local context for the rest of the crew."""

    role = "Research Engineer"
    goal = "Find relevant facts, previous experience and technical context."
    backstory = (
        "Specialist in collecting manufacturing, quality-control and domain knowledge "
        "for milk packaging closure defect analysis."
    )

    def __init__(
        self,
        message_bus,
        llm: Any | None = None,
        semantic_memory: SemanticMemory | None = None,
        search_tool: SearchWebTool | None = None,
    ) -> None:
        super().__init__(message_bus=message_bus, llm=llm)
        self.semantic_memory = semantic_memory or SemanticMemory(
            persist_directory="./chroma_store",
            collection_name="week3_research_memory",
        )
        self.search_tool = search_tool or SearchWebTool()

    def _build_research_brief(self, topic: str, defect_type: str, production_stage: str) -> str:
        memories = self.semantic_memory.search(topic, n_results=2)
        search_results = self.search_tool._run(query=topic, num_results=3)
        memory_block = "\n".join(memories) if memories else "Похожих записей в локальной памяти пока нет."
        return (
            f"Тема исследования: {topic}\n"
            f"Тип дефекта: {defect_type}\n"
            f"Этап производства: {production_stage}\n\n"
            "1. Найденный локальный контекст:\n"
            f"{memory_block}\n\n"
            "2. Внешние результаты поиска:\n"
            f"{search_results}\n\n"
            "3. Предварительный вывод исследователя:\n"
            "Нужно проверить, как связаны настройки укупорочного узла, качество крышек, "
            "скорость конвейера и статистика брака по партиям."
        )

    def process(self, payload: dict[str, Any]) -> AgentExecutionResult:
        topic = payload["topic"]
        defect_type = payload["defect_type"]
        production_stage = payload["production_stage"]
        research_brief = self._build_research_brief(topic, defect_type, production_stage)
        self.semantic_memory.add_memory(
            memory_id=f"research::{topic}",
            text=research_brief,
            metadata={"agent": self.agent_name, "topic": topic},
        )
        self.send_message(
            recipient="AnalystAgent",
            content=research_brief,
            metadata={"topic": topic, "stage": "research_complete"},
        )
        return AgentExecutionResult(
            agent_name=self.agent_name,
            content=research_brief,
            metadata={"topic": topic, "defect_type": defect_type},
        )
