"""Basic persistent-free world state and event context for TARA."""

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass(frozen=True)
class WorldEvent:
    """A normalized observation/event entering the world model."""

    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)


class WorldState:
    """Small explicit key/value world model with bounded event history."""

    def __init__(self, *, max_events: int = 128):
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self.values: dict[str, Any] = {}
        self.events: list[WorldEvent] = []
        self.max_events = max_events

    def apply(self, event: WorldEvent) -> WorldEvent:
        if not event.event_type:
            raise ValueError("event_type must not be empty")
        self.values.update(event.data)
        self.events.append(event)
        if len(self.events) > self.max_events:
            del self.events[:-self.max_events]
        return event

    def update(self, event_type: str, **data: Any) -> WorldEvent:
        return self.apply(WorldEvent(event_type, dict(data)))

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def recent(self, limit: int | None = None) -> tuple[WorldEvent, ...]:
        if limit is None:
            return tuple(self.events)
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return tuple(self.events[-limit:]) if limit else ()

    def snapshot(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class WorldContext:
    """Read-only context assembled for planning/reasoning."""

    state: dict[str, Any]
    recent_events: tuple[WorldEvent, ...]

    def as_prompt_context(self) -> str:
        state = ", ".join(f"{key}={value!r}" for key, value in sorted(self.state.items()))
        events = "; ".join(f"{event.event_type}: {event.data}" for event in self.recent_events)
        return f"World state: {state or 'empty'}\nRecent events: {events or 'none'}"


class WorldModel:
    """Facade for updating world state and producing bounded context."""

    def __init__(self, state: WorldState | None = None):
        self.state = state or WorldState()

    def observe(self, event_type: str, **data: Any) -> WorldEvent:
        return self.state.update(event_type, **data)

    def context(self, *, event_limit: int = 8) -> WorldContext:
        return WorldContext(self.state.snapshot(), self.state.recent(event_limit))
