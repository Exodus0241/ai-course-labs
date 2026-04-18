"""Writer agent for final packaging of the collaborative result."""

from __future__ import annotations

from src.agents.base_agent import AgentExecutionResult, BaseSpecializedAgent


class WriterAgent(BaseSpecializedAgent):
    """Assembles the final user-facing answer based on the previous agents."""

    role = "Technical Writer"
    goal = "Prepare a concise and structured final report for the user."
    backstory = (
        "Technical communicator who turns multi-agent outputs into a readable, "
        "well-structured summary for a lab report and demo."
    )

    def _extract_highlights(self, analysis: str) -> str:
        lines = [line.strip() for line in analysis.splitlines() if line.strip()]
        bullets = []
        for line in lines:
            if line.startswith("- "):
                bullets.append(line)
            if len(bullets) >= 4:
                break
        return "\n".join(bullets) if bullets else "- Ключевые выводы сформированы аналитиком."

    def process(self, payload: dict[str, object]) -> AgentExecutionResult:
        analysis = str(payload.get("analysis_context") or "\n\n".join(self.read_inbox()))
        topic = str(payload["topic"])
        highlights = self._extract_highlights(analysis)
        final_report = (
            "# Финальный отчёт crew\n\n"
            f"**Тема:** {topic}\n\n"
            "## Участники системы\n"
            "- `ResearcherAgent` собирает контекст, память и результаты поиска.\n"
            "- `AnalystAgent` выявляет причины дефекта и точки контроля.\n"
            "- `WriterAgent` оформляет итоговую сводку для пользователя.\n\n"
            "## Ключевые выводы\n"
            f"{highlights}\n\n"
            "## Подробный аналитический результат\n"
            f"{analysis}\n\n"
            "## Рекомендации для информационной системы\n"
            "- Добавить карточку инцидента с типом дефекта, партией и сменой.\n"
            "- Хранить параметры линии, фото дефекта и действия оператора.\n"
            "- Вести журнал межагентных сообщений и решений.\n"
            "- Показывать KPI по браку, повторяемости причин и динамике по партиям."
        )
        self.send_message(
            recipient="Supervisor",
            content=final_report,
            metadata={"stage": "writing_complete", "topic": topic},
        )
        return AgentExecutionResult(
            agent_name=self.agent_name,
            content=final_report,
            metadata={"topic": topic},
        )
