from src.goal_progress import GoalProgress
from src.reflection import Experience, ExperienceMemory, Reflector
from src.reflection_loop import ReflectionLoop


def test_experience_memory_keeps_recent_records(tmp_path):
    memory = ExperienceMemory(tmp_path / "exp.jsonl", max_records=2)
    memory.add(Experience("g", "a", "o1", True))
    memory.add(Experience("g", "b", "o2", False, feedback="failed"))
    memory.add(Experience("g", "c", "o3", True))
    assert [item.action for item in memory.records()] == ["b", "c"]
    memory.save()
    loaded = ExperienceMemory(memory.path, max_records=2)
    assert [item.action for item in loaded.records()] == ["b", "c"]


def test_reflector_reports_success_rate():
    reflector = Reflector()
    result = reflector.reflect(
        "g",
        [
            Experience("g", "a", "ok", True),
            Experience("g", "b", "bad", False, feedback="retry"),
        ],
    )
    assert result.attempts == 2
    assert result.successes == 1
    assert result.failures == 1
    assert result.success_rate == 0.5
    assert "retry" in result.lesson


def test_reflection_loop_records_and_reflects():
    loop = ReflectionLoop()
    cycle = loop.record("g", "tool", "done", True, score=1.0)
    assert cycle.reflection.success_rate == 1.0
    assert cycle.reflection.next_action


def test_goal_progress_is_bounded():
    progress = GoalProgress()
    progress.start("g", total=3)
    progress.mark_complete("g", 10)
    result = progress.snapshot("g")
    assert result.completed == 3
    assert result.ratio == 1.0
    assert result.status == "complete"
