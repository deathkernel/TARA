from src.tara_mind.memory.consolidation import ReplayBuffer, ReplayItem


def test_high_prediction_error_replays_first():
    buffer = ReplayBuffer(capacity=10)
    buffer.add(ReplayItem("easy", "easy", prediction_error=0.1))
    buffer.add(ReplayItem("surprise", "surprise", prediction_error=2.0))
    assert buffer.sample(1)[0].episode_id == "surprise"


def test_replay_count_reduces_priority():
    item = ReplayItem("x", "x", prediction_error=1.0)
    assert item.priority() > ReplayItem(
        "x", "x", prediction_error=1.0, replay_count=5
    ).priority()


def test_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=2)
    for i in range(5):
        buffer.add(ReplayItem(str(i), str(i), prediction_error=float(i)))
    assert len(buffer) == 2
