"""In-memory message bus used for explicit agent-to-agent communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Message:
    """A single communication event between two agents."""

    sender: str
    recipient: str
    content: str
    message_type: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class InMemoryMessageBus:
    """Stores, filters and summarizes messages exchanged by agents."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def publish(
        self,
        sender: str,
        recipient: str,
        content: str,
        message_type: str = "info",
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        message = Message(
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )
        self._messages.append(message)
        return message

    def get_messages_for(self, recipient: str) -> list[Message]:
        return [message for message in self._messages if message.recipient == recipient]

    def export_history(self) -> list[dict[str, Any]]:
        return [
            {
                "sender": message.sender,
                "recipient": message.recipient,
                "message_type": message.message_type,
                "content": message.content,
                "metadata": message.metadata,
                "timestamp": message.timestamp,
            }
            for message in self._messages
        ]
