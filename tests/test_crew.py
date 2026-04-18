"""Tests for the Week 3 multi-agent system."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.communication.message_bus import InMemoryMessageBus
from src.crew.research_crew import MultiAgentResearchCrew
from src.tasks.task_definitions import build_task_definitions


def build_payload() -> dict[str, str]:
    return {
        "topic": "Контроль дефектов закупорки молочной тары на линии розлива",
        "defect_type": "неполная закупорка",
        "production_stage": "укупорочный модуль",
        "symptoms": "смещение крышки, протечка и повторяющийся брак партии",
    }


def test_message_bus_delivers_messages() -> None:
    bus = InMemoryMessageBus()
    bus.publish("ResearcherAgent", "AnalystAgent", "brief", message_type="handoff")
    messages = bus.get_messages_for("AnalystAgent")
    assert len(messages) == 1
    assert messages[0].content == "brief"


def test_task_definitions_return_three_core_tasks() -> None:
    tasks = build_task_definitions(build_payload())
    assert len(tasks) == 3
    assert tasks[0].name == "research_task"


def test_crew_runs_end_to_end() -> None:
    crew = MultiAgentResearchCrew(
        llm=FakeListChatModel(
            responses=[
                "1. Краткое описание дефекта.\n2. Вероятные причины.\n3. Контрольные параметры."
            ]
        )
    )
    result = crew.kickoff(build_payload())
    assert "Финальный отчёт crew" in result.final_output
    assert len(result.messages) == 3
    assert result.messages[0]["recipient"] == "AnalystAgent"


def test_crew_result_is_serializable() -> None:
    crew = MultiAgentResearchCrew(
        llm=FakeListChatModel(
            responses=[
                "1. Краткое описание дефекта.\n2. Вероятные причины.\n3. Контрольные параметры."
            ]
        )
    )
    result = crew.kickoff_as_dict(build_payload())
    assert result["topic"]
    assert isinstance(result["messages"], list)
