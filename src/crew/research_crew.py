"""CrewAI-compatible orchestration for the Week 3 multi-agent system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    from crewai import Crew, Process, Task
except Exception:  # pragma: no cover - optional dependency
    Crew = None
    Process = None
    Task = None

from src.agent_core import Week3AgentRuntime
from src.agents.analyst_agent import AnalystAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.writer_agent import WriterAgent
from src.communication.message_bus import InMemoryMessageBus
from src.tasks.task_definitions import build_task_definitions


@dataclass
class CrewRunResult:
    """Full output of the multi-agent execution."""

    topic: str
    research_output: str
    analysis_output: str
    final_output: str
    messages: list[dict[str, Any]]
    used_crewai: bool


class MultiAgentResearchCrew:
    """Coordinates three specialized agents around a shared production case."""

    def __init__(self, llm: Any | None = None) -> None:
        runtime = Week3AgentRuntime(llm=llm)
        self.llm = runtime.llm
        self.message_bus = InMemoryMessageBus()
        self.researcher = ResearcherAgent(
            message_bus=self.message_bus,
            llm=self.llm,
            semantic_memory=runtime.semantic_memory,
        )
        self.analyst = AnalystAgent(message_bus=self.message_bus, llm=self.llm)
        self.writer = WriterAgent(message_bus=self.message_bus, llm=self.llm)

    def build_crewai_crew(self, payload: dict[str, Any]) -> Crew | None:
        if Crew is None or Task is None or Process is None:
            return None

        researcher_agent = self.researcher.create_crewai_agent()
        analyst_agent = self.analyst.create_crewai_agent()
        writer_agent = self.writer.create_crewai_agent()
        if not all([researcher_agent, analyst_agent, writer_agent]):
            return None

        task_specs = build_task_definitions(payload)
        crew_tasks = [
            Task(
                description=task_specs[0].description,
                expected_output=task_specs[0].expected_output,
                agent=researcher_agent,
            ),
            Task(
                description=task_specs[1].description,
                expected_output=task_specs[1].expected_output,
                agent=analyst_agent,
            ),
            Task(
                description=task_specs[2].description,
                expected_output=task_specs[2].expected_output,
                agent=writer_agent,
            ),
        ]
        return Crew(
            agents=[researcher_agent, analyst_agent, writer_agent],
            tasks=crew_tasks,
            process=Process.sequential,
            verbose=True,
        )

    def kickoff(self, payload: dict[str, Any]) -> CrewRunResult:
        research = self.researcher.process(payload)
        analyst_payload = dict(payload)
        analyst_payload["research_context"] = research.content
        analysis = self.analyst.process(analyst_payload)
        writer_payload = dict(payload)
        writer_payload["analysis_context"] = analysis.content
        final = self.writer.process(writer_payload)

        used_crewai = self.build_crewai_crew(payload) is not None
        return CrewRunResult(
            topic=payload["topic"],
            research_output=research.content,
            analysis_output=analysis.content,
            final_output=final.content,
            messages=self.message_bus.export_history(),
            used_crewai=used_crewai,
        )

    def kickoff_as_dict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return asdict(self.kickoff(payload))
