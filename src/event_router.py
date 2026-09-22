"""Basic event routing for TARA world observations."""

from dataclasses import dataclass
from typing import Callable

from .world_state import WorldEvent, WorldModel


@dataclass(frozen=True)
class EventRule:
    event_type: str
    action: Callable[[WorldEvent], object]


class EventRouter:
    """Route selected world events to explicit host-provided actions."""

    def __init__(self, world: WorldModel | None = None):
        self.world = world or WorldModel()
        self._rules: dict[str, list[Callable[[WorldEvent], object]]] = {}

    def on(self, event_type: str, action: Callable[[WorldEvent], object]) -> None:
        if not event_type:
            raise ValueError("event_type must not be empty")
        self._rules.setdefault(event_type, []).append(action)

    def emit(self, event_type: str, **data) -> WorldEvent:
        event = self.world.observe(event_type, **data)
        for action in tuple(self._rules.get(event_type, ())):
            action(event)
        return event

    def clear(self, event_type: str | None = None) -> None:
        if event_type is None:
            self._rules.clear()
        else:
            self._rules.pop(event_type, None)
