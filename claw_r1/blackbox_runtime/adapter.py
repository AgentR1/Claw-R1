"""The only black-box-agent extension contract owned by Claw-R1 core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BlackBoxAdapter(ABC):
    """Recipe-owned implementation of one external black-box agent.

    Core owns the sandbox, session, Gateway route and idempotent turn store.
    A recipe owns how its agent is configured, started, resumed and parsed.
    """

    @abstractmethod
    async def run(self, state: dict[str, Any], messages: list[dict[str, Any]], turn_id: str) -> dict[str, Any]:
        """Run one agent turn in the supplied isolated workspace."""
        raise NotImplementedError
