from src.event_router import EventRouter
from src.world_state import WorldModel, WorldState


def test_world_state_applies_events_and_keeps_bounded_history():
    state = WorldState(max_events=2)
    state.update("system", status="ready")
    state.update("network", online=True)
    state.update("system", status="busy")

    assert state.get("status") == "busy"
    assert state.get("online") is True
    assert len(state.recent()) == 2


def test_world_context_is_read_only_snapshot():
    world = WorldModel()
    world.observe("location", city="Pune")
    context = world.context()
    context.state["city"] = "other"

    assert world.state.get("city") == "Pune"
    assert "Pune" in context.as_prompt_context()


def test_event_router_updates_world_and_calls_explicit_actions():
    router = EventRouter()
    seen = []
    router.on("alarm", lambda event: seen.append(event.data["level"]))

    router.emit("alarm", level="high")

    assert seen == ["high"]
    assert router.world.state.get("level") == "high"
