"""Integrated cognitive brain for TARA."""
from dataclasses import dataclass
import random
from pathlib import Path
from .advanced_planning import AdaptivePlan, AdvancedPlanner, PlanAlternative, PlanStep
from .algorithm_synthesis import AlgorithmArchive, AlgorithmDiscoveryEngine, AlgorithmProposal, DiscoveryCycle, ModelAlgorithmGenerator, VerificationReport
from .agent import AgentLoop
from .autonomous_orchestrator import AutonomousOrchestrator, TaskNode
from .continual_learning_engine import ContinualLearningEngine, PromotionResult, ReplayBatch
from .multimodal_perception import MultimodalContext, MultimodalObservation, MultimodalPerception
from .autonomous_research import AutonomousResearchEngine, ResearchReport
from .scientific_experiment import ExperimentDesign, ExperimentReport, ScientificExperimentEngine
from .architecture_optimization import ArchitectureOptimizer, ArchitectureVariant, OptimizationResult
from .experience_learning import ExperienceLearningEngine
from .intelligence_benchmark import BenchmarkReport, IntelligenceBenchmark
from .intelligence_improvement import ImprovementProposal, IntelligenceImprovementEngine, IntelligenceImprovementReport
from .unified_cognitive_loop import ActionOutcome, ActionRequest, CognitiveLoopReport, UnifiedCognitiveLoop
from .event_router import EventRouter
from .goal_progress import GoalProgress
from .integration import TARAEngine
from .memory import LongTermMemory
from .reflection_loop import ReflectionLoop
from .self_improvement_lab import ImprovementCycle, SelfImprovementLab
from .structured_reasoning import ReasoningState, StructuredReasoner
from .temporal_perception import TemporalContext, TemporalPerception
from .tool_intelligence import ToolAttempt, ToolCapability, ToolIntelligence
from .world_state import WorldContext, WorldModel

@dataclass(frozen=True)
class BrainResponse:
    text: str
    recalled: tuple = ()
    observations: tuple = ()
    temporal_context: str = ""

class TARABrain:
    def __init__(self, model, tokenizer=None, *, engine=None, memory=None, seed=0, reflection=None,
                 progress=None, orchestrator=None, world=None, events=None, temporal=None,
                 temporal_history=256, reasoner=None, planner=None, tools=None, improver=None,
                 continual=None, multimodal=None, researcher=None, experimenter=None, architecture_optimizer=None, cognitive_loop=None, experience_learning=None,
                 intelligence_improver=None):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = TARAEngine() if engine is None else engine
        self.memory = LongTermMemory() if memory is None else memory
        self.agent = AgentLoop(engine=self.engine, memory=self.memory)
        self.reflection = reflection or ReflectionLoop()
        self.progress = progress or GoalProgress()
        self.orchestrator = orchestrator or AutonomousOrchestrator()
        self.world = world or WorldModel()
        self.events = events or EventRouter(self.world)
        self.temporal = temporal or TemporalPerception(max_history=temporal_history)
        self.reasoner = reasoner or StructuredReasoner()
        self.planner = planner or AdvancedPlanner()
        self.tools = tools or ToolIntelligence()
        self.improver = improver or SelfImprovementLab()
        self.continual = continual or ContinualLearningEngine(seed=seed)
        self.multimodal = multimodal or MultimodalPerception()
        self.researcher = researcher or AutonomousResearchEngine()
        self.experimenter = experimenter or ScientificExperimentEngine()
        self.architecture_optimizer = architecture_optimizer
        self.experience_learning = experience_learning or ExperienceLearningEngine()
        self.intelligence_improver = intelligence_improver or IntelligenceImprovementEngine()
        self.cognitive_loop = cognitive_loop or UnifiedCognitiveLoop(
            reasoner=self.reasoner, planner=self.planner, tools=self.tools, memory=self.memory,
            reflection=self.reflection, progress=self.progress, learning=self.experience_learning,
        )
        self.rng = random.Random(seed)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, engine=None, memory=None, seed=0):
        from .model_runtime import load_checkpoint
        model, tokenizer = load_checkpoint(path)
        return cls(model, tokenizer, engine=engine, memory=memory, seed=seed)

    def build_replay(self, records) -> ReplayBatch: return self.continual.build_replay(records)
    def perceive_multimodal(self, observations: list[MultimodalObservation]) -> MultimodalContext: return self.multimodal.perceive(observations)
    def perceive_text(self, text: str, *, confidence=1.0) -> MultimodalObservation: return self.multimodal.text(text, confidence=confidence)
    def perceive_document(self, text: str, *, path="", confidence=1.0) -> MultimodalObservation: return self.multimodal.document(text, path=path, confidence=confidence)
    def perceive_image(self, data: bytes, *, confidence=0.8) -> MultimodalObservation: return self.multimodal.image(data, confidence=confidence)
    def perceive_audio(self, data: bytes, *, confidence=0.7) -> MultimodalObservation: return self.multimodal.audio(data, confidence=confidence)
    def perceive_screen(self, elements, *, confidence=0.9) -> MultimodalObservation: return self.multimodal.screen(elements, confidence=confidence)
    def conduct_research(self, question: str, searcher, *, max_queries=None) -> ResearchReport: return self.researcher.research(question, searcher, max_queries=max_queries)
    def run_experiment(self, design: ExperimentDesign, executor) -> ExperimentReport: return self.experimenter.run(design, executor)
    def discover_algorithm(self, problem, *, generator, verifier, rounds=4, candidates_per_round=8, archive=None) -> DiscoveryCycle:
        """Invent, verify, refine, and retain algorithm hypotheses without executing generated code."""
        engine = AlgorithmDiscoveryEngine(generator, verifier, archive=archive)
        return engine.discover(problem, rounds=rounds, candidates_per_round=candidates_per_round)

    def optimize_architecture(self, baseline: ArchitectureVariant, *, rounds=3, candidates_per_round=4, optimizer: ArchitectureOptimizer | None = None) -> OptimizationResult:
        engine = optimizer or self.architecture_optimizer
        if engine is None: raise ValueError("an ArchitectureOptimizer with an injected evaluator is required")
        return engine.optimize(baseline, rounds=rounds, candidates_per_round=candidates_per_round)
    def benchmark_intelligence(self, benchmark: IntelligenceBenchmark, solver) -> BenchmarkReport: return benchmark.run(solver)
    def improve_intelligence(self, baseline: BenchmarkReport, runner) -> IntelligenceImprovementReport: return self.intelligence_improver.improve(baseline, runner)
    def propose_intelligence_improvement(self, baseline: BenchmarkReport) -> ImprovementProposal:
        return self.intelligence_improver.planner.propose(self.intelligence_improver.analyzer.analyze(baseline))
    def start_cognitive_loop(self, goal: str, subtasks, *, total=None): return self.cognitive_loop.start(goal, subtasks, total=total)
    def cognitive_cycle(self, perception, *, action_executor, expected=None, required_capabilities=(), tool_facts=None): return self.cognitive_loop.cycle(perception, action_executor=action_executor, expected=expected, required_capabilities=required_capabilities, tool_facts=tool_facts)
    def run_cognitive_loop(self, goal: str, subtasks, perceptions, *, action_executor, expected=None, required_capabilities=(), tool_facts=None) -> CognitiveLoopReport: return self.cognitive_loop.run(goal, subtasks, perceptions, action_executor=action_executor, expected=expected, required_capabilities=required_capabilities, tool_facts=tool_facts)
    def stop_cognitive_loop(self): self.cognitive_loop.stop()
    def resume_cognitive_loop(self): self.cognitive_loop.resume()
    def evaluate_learning(self, baseline, current) -> PromotionResult: return self.continual.evaluate(baseline, current)

    def observe(self, observation, *, remember_key=None, importance=1.0, confidence=1.0, source="agent", kind="observation", timestamp=None):
        result = self.agent.observe(observation, remember_key=remember_key, importance=importance)
        normalized = self.temporal.ingest(observation, source=source, kind=kind, confidence=confidence, timestamp=timestamp)
        self.events.emit("observation", value=observation, observation_id=normalized.observation_id, confidence=normalized.confidence, timestamp=normalized.timestamp)
        return result
    def observe_perception(self, observation, *, source="perception", kind="observation", confidence=1.0, timestamp=None, remember_key=None, importance=1.0):
        normalized = self.temporal.ingest(observation, source=source, kind=kind, confidence=confidence, timestamp=timestamp)
        self.agent.observe(normalized.content, remember_key=remember_key, importance=importance)
        self.events.emit("perception", observation_id=normalized.observation_id, source=normalized.source, kind=normalized.kind, content=normalized.content, confidence=normalized.confidence, timestamp=normalized.timestamp)
        return normalized
    def temporal_context(self, *, limit=16, min_confidence=0.0) -> TemporalContext: return self.temporal.context(limit=limit, min_confidence=min_confidence)
    def reason(self, goal, *, facts=(), assumptions=(), hypotheses=(), evidence=(), required_claim_ids=()):
        state = self.reasoner.start(goal); return self.reasoner.reason(state, facts=facts, assumptions=assumptions, hypotheses=hypotheses, evidence=evidence, required_claim_ids=required_claim_ids)
    def verify_reasoning(self, state: ReasoningState, *, required_claim_ids=()): return self.reasoner.verifier.verify(state, required_claim_ids=required_claim_ids)
    def create_plan(self, goal, subtasks, *, max_cost=None) -> AdaptivePlan:
        planner = self.planner if max_cost is None else AdvancedPlanner(max_cost=max_cost); plan = planner.create(goal, subtasks); self.engine.set_goal(goal); self.engine.state.plan = None; return plan
    def validate_plan(self, plan: AdaptivePlan): return self.planner.validate(plan)
    def plan_alternatives(self, plan: AdaptivePlan, *, count=3) -> tuple[PlanAlternative, ...]: return self.planner.alternatives(plan, count=count)
    def revise_plan(self, plan: AdaptivePlan, failed_step_id, feedback, replacement: PlanStep | None = None): return self.planner.revise(plan, failed_step_id, feedback, replacement)
    def register_tool_capability(self, capability: ToolCapability) -> None: self.tools.register(capability)
    def choose_tool(self, goal: str, required_capabilities, *, facts=None, preferred_latency=None, risk_tolerance=.5): return self.tools.choose(goal, required_capabilities, facts=facts, preferred_latency=preferred_latency, risk_tolerance=risk_tolerance)
    def recover_tool(self, attempt: ToolAttempt, goal: str, required_capabilities, *, facts=None, risk_tolerance=.5): return self.tools.recover(attempt, goal, required_capabilities, facts=facts, risk_tolerance=risk_tolerance)
    def improve_candidate(self, baseline, problem, *, rounds=3, mutation_count=4) -> tuple: return self.improver.iterate(baseline, problem, rounds=rounds, mutation_count=mutation_count)
    def improvement_cycle(self, baseline, problem, *, mutation_count=4) -> ImprovementCycle: return self.improver.cycle(baseline, problem, mutation_count=mutation_count)
    def observe_world(self, event_type, **data): return self.events.emit(event_type, **data)
    def world_context(self, *, event_limit=8) -> WorldContext: return self.world.context(event_limit=event_limit)
    def set_goal(self, description, success_conditions=()): return self.engine.set_goal(description, success_conditions)
    def plan(self, subtasks): return self.engine.plan(subtasks)
    def start_progress(self, goal, total=1): return self.progress.start(goal, total)
    def mark_progress(self, goal, count=1): return self.progress.mark_complete(goal, count)
    def record_experience(self, goal, action, outcome, success, score=0.0, feedback=""): return self.reflection.record(goal, action, outcome, success, score, feedback)
    def reflect(self, goal): return self.reflection.reflect(goal)
    def add_task(self, task_id, description, *, priority=0, dependencies=(), retries=0):
        task = TaskNode(task_id, description, priority, tuple(dependencies), retries); self.orchestrator.queue.add(task); return task
    def run_tasks(self, executor, *, max_steps=None):
        if max_steps is not None:
            if max_steps <= 0: raise ValueError("max_steps must be positive")
            self.orchestrator.max_steps = max_steps
        return self.orchestrator.run(executor)
    def stop_tasks(self): self.orchestrator.stop()
    def resume_tasks(self): self.orchestrator.resume()
    def recall(self, query, limit=None, min_score=0.0): return self.agent.recall(query, limit=limit, min_score=min_score)
    def generate(self, prompt, *, max_new_tokens=32, temperature=1.0, top_k=None, top_p=None):
        if self.tokenizer is None: raise ValueError("a tokenizer is required for text generation")
        if max_new_tokens <= 0 or temperature <= 0: raise ValueError("max_new_tokens must be positive and temperature must be positive")
        if top_p is not None and not 0.0 < top_p <= 1.0: raise ValueError("top_p must be in (0, 1]")
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids: raise ValueError("prompt must encode to at least one token")
        for _ in range(max_new_tokens):
            token_id = self.model.sample_next_token(token_ids, temperature=temperature, top_k=top_k, rng=self.rng) if top_p is None else self._sample_top_p(token_ids, temperature, top_p); token_ids.append(token_id)
        return self.tokenizer.decode(token_ids)
    def _sample_top_p(self, token_ids, temperature, top_p):
        import math
        logits = list(self.model.forward_numeric(token_ids)[-1]); scaled = [value / temperature for value in logits]; maximum = max(scaled); probabilities = [math.exp(value - maximum) for value in scaled]; total = sum(probabilities); probabilities = [value / total for value in probabilities]; ranked = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True); selected=[]; cumulative=0.0
        for index in ranked:
            selected.append(index); cumulative += probabilities[index]
            if cumulative >= top_p: break
        threshold = self.rng.random() * sum(probabilities[index] for index in selected); cumulative=0.0
        for index in selected:
            cumulative += probabilities[index]
            if threshold < cumulative: return index
        return selected[-1]
    def build_reasoning_context(self, prompt, *, memory_limit=8, event_limit=8, temporal_limit=16, min_confidence=0.0):
        recalled = tuple(self.memory.retrieve(prompt, limit=memory_limit)); world=self.world_context(event_limit=event_limit); temporal=self.temporal_context(limit=temporal_limit, min_confidence=min_confidence); memory_text="
".join(str(item) for item in recalled) or "none"; return f"User/task: {prompt}

Memory:
{memory_text}

{world.as_prompt_context()}

{temporal.as_prompt_context()}"
    def respond(self, prompt, *, remember_key=None, importance=1.0, max_new_tokens=32, temperature=1.0, top_k=None, top_p=None):
        self.observe(prompt, remember_key=remember_key, importance=importance, source="user", kind="prompt"); recalled=tuple(self.memory.retrieve(prompt)); context=self.build_reasoning_context(prompt)
        text = self.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
        return BrainResponse(text=text, recalled=recalled, observations=tuple(self.engine.state.observations.recent()), temporal_context=self.temporal_context().as_prompt_context())
    def verify_result(self, observed, expected): return self.agent.submit_result(observed, expected)
    def recover(self, replacement_tasks): return self.agent.recover(replacement_tasks)
