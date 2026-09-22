import pytest

from src.advanced_planning import AdvancedPlanner, PlanStep, PlanValidator, StepStatus


def test_goal_decomposition_builds_dependency_plan():
    planner = AdvancedPlanner()
    plan = planner.create("ship feature", ["design", "implement", "verify"])
    assert plan.steps["step-1"].dependencies == ("step-0",)
    assert plan.ready_steps()[0].description == "design"


def test_validator_detects_dependency_cycles_and_budget_overruns():
    plan = planner = AdvancedPlanner().decomposer.decompose(
        "cycle", [
            {"step_id": "a", "description": "A", "dependencies": ["b"], "cost": 3},
            {"step_id": "b", "description": "B", "dependencies": ["a"], "cost": 3},
        ]
    )
    result = PlanValidator().validate(plan, max_cost=5)
    assert not result.valid
    assert any("cycle" in error for error in result.errors)
    assert any("budget" in error for error in result.errors)


def test_alternatives_and_revision_preserve_completed_work():
    planner = AdvancedPlanner()
    plan = planner.create("recover", ["prepare", "execute", "verify"])
    plan.steps["step-0"].status = StepStatus.COMPLETED
    alternatives = planner.alternatives(plan)
    assert len(alternatives) == 3
    replacement = PlanStep("replacement", "alternative execute", ("step-0",), 1.0, 2)
    planner.revise(plan, "step-1", "execution failed", replacement)
    assert plan.steps["step-0"].status == StepStatus.COMPLETED
    assert plan.revision == 1
    assert plan.steps["step-2"].dependencies == ("replacement",)
