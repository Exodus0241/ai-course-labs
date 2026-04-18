"""Base abstractions for specialized multi-agent workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

try:
    from crewai import Agent
except Exception:  # pragma: no cover - optional dependency
    Agent = None

from src.communication.message_bus import InMemoryMessageBus


@dataclass
class AgentExecutionResult:
    """Normalized output of any specialized agent."""

    agent_name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSpecializedAgent(ABC):
    """Shared behavior for domain-focused agents."""

    role: str = "Specialized agent"
    goal: str = "Solve a delegated task"
    backstory: str = "A collaborative AI worker."

    def __init__(self, message_bus: InMemoryMessageBus, llm: Any | None = None) -> None:
        self.message_bus = message_bus
        self.llm = llm

    @property
    def agent_name(self) -> str:
        return self.__class__.__name__

    def create_crewai_agent(self) -> Agent | None:
        if Agent is None:
            return None
        try:
            return Agent(
                role=self.role,
                goal=self.goal,
                backstory=self.backstory,
                llm=self.llm,
                allow_delegation=True,
                verbose=True,
            )
        except Exception:
            return None

    def send_message(
        self,
        recipient: str,
        content: str,
        message_type: str = "handoff",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.message_bus.publish(
            sender=self.agent_name,
            recipient=recipient,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )

    def read_inbox(self) -> list[str]:
        return [message.content for message in self.message_bus.get_messages_for(self.agent_name)]

    @abstractmethod
    def process(self, payload: dict[str, Any]) -> AgentExecutionResult:
        """Executes the agent's domain task."""
