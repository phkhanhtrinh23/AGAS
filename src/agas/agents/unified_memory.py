"""Shared unified memory for single-LLM coordinator+worker architecture.

When ``--unified-memory`` is enabled, one LLM acts as both the coordinator and
every worker within a step. They share a single bounded deque of recent events
instead of maintaining per-agent trajectory summaries. Every role assignment
and every worker action appends to this log; both prompt builders read the
same snapshot.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List


class UnifiedMemory:
    """Bounded shared memory of recent coordinator+worker events."""

    def __init__(self, maxlen: int = 15) -> None:
        """Initialize the memory with a maximum length.

        Args:
            maxlen: Maximum number of recent events retained.
        """

        self._events: deque[Dict[str, Any]] = deque(maxlen=int(maxlen))
        self.maxlen: int = int(maxlen)

    def append(self, event: Dict[str, Any]) -> None:
        """Append an event; oldest entries are discarded past ``maxlen``.

        Args:
            event: Dictionary describing the event (actor/kind/payload/step).
        """

        self._events.append(dict(event))

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return the current memory contents as a plain list.

        Returns:
            List of event dicts in insertion order.
        """

        return list(self._events)

    def __len__(self) -> int:
        """Return the number of events currently stored."""

        return len(self._events)
