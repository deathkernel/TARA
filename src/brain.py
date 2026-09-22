"""Integrated, PC-free cognitive brain for TARA.

The brain is the final pre-tool integration boundary: perception, memory,
language generation, reasoning and verification are coordinated here, while
external actions remain outside the module.  This keeps the model complete
before any future tool/PC execution layer is attached.
"""

from dataclasses import dataclass
import random
from pathlib import Path

from .agent import AgentLoop
from .integration import TARAEngine
from .memory import LongTermMemory
from .tokenizer import CharTokenizer


@dataclass(frozen=True)
class BrainResponse:
    """A generated response plus the cognitive context used to produce it."""

    text: str
    recalled: tuple = ()
    observations: tuple = ()


class TARABrain:
    """Connect TARA's learned language core to its explicit cognitive layers."""

    def __init__(self, model, tokenizer=None, *, engine=None, memory=None, seed=0):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = TARAEngine() if engine is None else engine
        self.memory = LongTermMemory() if memory is None else memory
        self.agent = AgentLoop(engine=self.engine, memory=self.memory)
        self.rng = random.Random(seed)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, engine=None, memory=None, seed=0):
        """Build a brain directly from a trained PyTorch TARA checkpoint."""
        from .model_runtime import load_checkpoint

        model, tokenizer = load_checkpoint(path)
        return cls(model, tokenizer, engine=engine, memory=memory, seed=seed)

    def observe(self, observation, *, remember_key=None, importance=1.0):
        return self.agent.observe(
            observation,
            remember_key=remember_key,
            importance=importance,
        )

    def set_goal(self, description, success_conditions=()):
        return self.engine.set_goal(description, success_conditions)

    def plan(self, subtasks):
        return self.engine.plan(subtasks)

    def recall(self, query, limit=None, min_score=0.0):
        return self.agent.recall(query, limit=limit, min_score=min_score)

    def generate(self, prompt, *, max_new_tokens=32, temperature=1.0, top_k=None, top_p=None):
        """Generate text through the model's non-autograd inference path."""
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required for text generation")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_p is not None and not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("prompt must encode to at least one token")

        for _ in range(max_new_tokens):
            if top_p is None:
                token_id = self.model.sample_next_token(
                    token_ids,
                    temperature=temperature,
                    top_k=top_k,
                    rng=self.rng,
                )
            else:
                token_id = self._sample_top_p(token_ids, temperature, top_p)
            token_ids.append(token_id)
        return self.tokenizer.decode(token_ids)

    def _sample_top_p(self, token_ids, temperature, top_p):
        import math

        logits = list(self.model.forward_numeric(token_ids)[-1])
        scaled = [value / temperature for value in logits]
        maximum = max(scaled)
        probabilities = [math.exp(value - maximum) for value in scaled]
        total = sum(probabilities)
        probabilities = [value / total for value in probabilities]
        ranked = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)
        selected = []
        cumulative = 0.0
        for index in ranked:
            selected.append(index)
            cumulative += probabilities[index]
            if cumulative >= top_p:
                break
        threshold = self.rng.random() * sum(probabilities[index] for index in selected)
        cumulative = 0.0
        for index in selected:
            cumulative += probabilities[index]
            if threshold < cumulative:
                return index
        return selected[-1]

    def respond(self, prompt, *, remember_key=None, importance=1.0, max_new_tokens=32,
                temperature=1.0, top_k=None, top_p=None):
        """Observe a prompt, retrieve related memory, then generate a response."""
        self.observe(prompt, remember_key=remember_key, importance=importance)
        recalled = tuple(self.memory.retrieve(prompt))
        text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return BrainResponse(
            text=text,
            recalled=recalled,
            observations=tuple(self.engine.state.observations.recent()),
        )

    def verify_result(self, observed, expected):
        return self.agent.submit_result(observed, expected)

    def recover(self, replacement_tasks):
        return self.agent.recover(replacement_tasks)
