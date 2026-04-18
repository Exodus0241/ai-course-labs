"""Short-term memory implementation for the Week 3 project."""

from __future__ import annotations

from collections import deque
from typing import Deque


class WorkingMemory:
    """Stores the latest turns to preserve the local conversation context."""

    def __init__(self, capacity: int = 6) -> None:
        self.capacity = capacity
        self._items: Deque[str] = deque(maxlen=capacity)

    def add(self, item: str) -> None:
        self._items.append(item)

    def get_context(self) -> str:
        return "\n".join(self._items)

    def clear(self) -> None:
        self._items.clear()
