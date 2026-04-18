"""Analyst agent for turning research into structured conclusions."""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentExecutionResult, BaseSpecializedAgent
from src.tools.custom_tool import MilkClosureDefectTool


class AnalystAgent(BaseSpecializedAgent):
    """Produces a structured operational analysis for the final report."""

    role = "Quality Analyst"
    goal = "Transform collected data into practical production insights."
    backstory = (
        "Manufacturing analyst focused on dairy packaging lines, root cause analysis "
        "and digital quality-control systems."
    )

    def __init__(self, message_bus, llm: Any | None = None) -> None:
        super().__init__(message_bus=message_bus, llm=llm)
        self.defect_tool = MilkClosureDefectTool(llm=llm)

    def _build_analysis(self, defect_type: str, production_stage: str, symptoms: str, research: str) -> str:
        tool_output = self.defect_tool._run(
            defect_type=defect_type,
            production_stage=production_stage,
            symptoms=symptoms,
        )
        return (
            "## Аналитический отчёт\n\n"
            "### 1. Материал от исследователя\n"
            f"{research}\n\n"
            "### 2. Экспертный анализ дефекта\n"
            f"{tool_output}\n\n"
            "### 3. Решение аналитика\n"
            "- Контролировать момент укупорки и геометрию подачи крышки.\n"
            "- Отслеживать повторяемость брака по сменам, партиям и линии.\n"
            "- Добавить фотофиксацию и журнал инцидентов для последующего анализа.\n"
            "- Сохранять параметры оборудования рядом с карточкой дефекта."
        )

    def process(self, payload: dict[str, Any]) -> AgentExecutionResult:
        research = payload.get("research_context") or "\n\n".join(self.read_inbox())
        analysis = self._build_analysis(
            defect_type=payload["defect_type"],
            production_stage=payload["production_stage"],
            symptoms=payload["symptoms"],
            research=research,
        )
        self.send_message(
            recipient="WriterAgent",
            content=analysis,
            metadata={"stage": "analysis_complete", "defect_type": payload["defect_type"]},
        )
        return AgentExecutionResult(
            agent_name=self.agent_name,
            content=analysis,
            metadata={"symptoms": payload["symptoms"]},
        )
