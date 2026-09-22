"""Basic observe -> evaluate -> reflect -> adjust loop for TARA."""

from dataclasses import dataclass

from .reflection import Experience, ExperienceMemory, Reflector


@dataclass(frozen=True)
class ReflectionCycle:
    experience: Experience
    reflection: object


class ReflectionLoop:
    """Connect outcome recording to reflection and future-action feedback."""

    def __init__(self, memory=None, reflector=None):
        self.memory = memory or ExperienceMemory()
        self.reflector = reflector or Reflector()

    def record(self, goal, action, outcome, success, score=0.0, feedback=""):
        experience = Experience(goal, action, outcome, bool(success), float(score), feedback)
        self.memory.add(experience)
        reflection = self.reflector.reflect(goal, self.memory.records(goal))
        return ReflectionCycle(experience, reflection)

    def reflect(self, goal):
        return self.reflector.reflect(goal, self.memory.records(goal))
