"""Short-term memory implementation for the agent."""

from __future__ import annotations

from collections import deque
from typing import Deque


class WorkingMemory:
    """Stores the latest turns to preserve the local conversation context."""

    def __init__(self, capacity: int = 6) -> None:
        self.capacity = capacity
        self._items: Deque[str] = deque(maxlen=capacity)

    def add(self, item: str) -> None:
        """Adds a new memory record to the rolling buffer."""
        self._items.append(item)

    def get_context(self) -> str:
        """Returns the latest messages as one compact context block."""
        return "\n".join(self._items)

    def clear(self) -> None:
        """Resets the short-term memory."""
        self._items.clear()
