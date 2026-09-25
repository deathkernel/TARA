from src.tara_mind.cognition.working_memory import BoundedWorkingMemory


def test_working_memory_never_exceeds_capacity():
    wm = BoundedWorkingMemory(capacity=3)
    for i in range(10):
        wm.add(i, priority=0.5)
    assert len(wm.slots) == 3


def test_high_priority_item_is_retained_over_old_low_priority_item():
    wm = BoundedWorkingMemory(capacity=2)
    wm.add("important", priority=1.0)
    wm.add("low", priority=0.0)
    wm.tick()
    wm.tick()
    wm.add("new", priority=1.0)
    assert any(slot.content == "important" for slot in wm.slots)


def test_attend_refreshes_age_and_increments_use():
    wm = BoundedWorkingMemory(capacity=2)
    idx = wm.add("memory", priority=0.5)
    wm.tick()
    wm.attend(idx)
    slot = wm.slots[idx]
    assert slot.age == 0
    assert slot.access_count == 1


def test_retention_is_ordered():
    wm = BoundedWorkingMemory(capacity=3)
    wm.add("a", priority=1.0)
    wm.add("b", priority=0.5)
    wm.add("c", priority=0.0)
    retained = wm.retain()
    assert retained[0].content == "a"


def test_invalid_capacity_is_rejected():
    try:
        BoundedWorkingMemory(capacity=0)
    except ValueError:
        return
    raise AssertionError("invalid capacity should be rejected")
