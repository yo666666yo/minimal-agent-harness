"""A small, reproducible two-agent environment for credit-assignment smoke tests.

The environment is deliberately model-free.  A coordinator and a researcher
read different sources concurrently, persist their findings through an
exclusive writer, and let the coordinator persist the final artifact.  The
policy is a Bernoulli read policy so that single-agent, team-GRPO, and a small
observable-q CAD-GRPO proxy can be compared without an API key or a model.

This is an experiment/data-flow harness, not evidence about a real LLM policy.
The q_i signal is observable confidence supplied by the synthetic workers, and
the oracle is a leave-one-agent-out marginal on the synthetic outcome reward.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import pathlib
import random
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence


AGENT_IDS = ("coordinator", "researcher")
METHODS = ("single-agent", "naive-grpo", "cad-grpo")
MODEL_NAME = "synthetic-read-policy-v1"
TOPOLOGY = "coordinator+researcher/read-parallel-write-serial"
VERIFIER_VERSION = "required-facts-v1"
ORACLE_VERSION = "leave-one-agent-out-total-marginal-v1"
SUCCESS_BONUS = 0.25

READ_TOKEN_COST = 24
SKIP_TOKEN_COST = 2
MESSAGE_TOKEN_COST = 10
FINDING_WRITE_TOKEN_COST = 12
FINAL_WRITE_TOKEN_COST = 20

READ_INPUT_TOKENS = 8
READ_OUTPUT_TOKENS = 16
SKIP_INPUT_TOKENS = 2
MESSAGE_INPUT_TOKENS = 4
MESSAGE_OUTPUT_TOKENS = 6
FINDING_WRITE_INPUT_TOKENS = 6
FINDING_WRITE_OUTPUT_TOKENS = 6
FINAL_WRITE_INPUT_TOKENS = 10
FINAL_WRITE_OUTPUT_TOKENS = 10


@dataclass(frozen=True)
class TaskSpec:
    """One task with one required fact per agent."""

    task_id: str
    prompt: str
    coordinator_source: str
    researcher_source: str
    coordinator_fact: str
    researcher_fact: str
    coordinator_value: float = 0.5
    researcher_value: float = 0.5

    def sources(self) -> dict[str, str]:
        return {
            self.coordinator_source: (
                f"Brief for {self.task_id}. Required context: "
                f"{self.coordinator_fact}."
            ),
            self.researcher_source: (
                f"Evidence for {self.task_id}. Required check: "
                f"{self.researcher_fact}."
            ),
        }

    def value_for(self, agent_id: str) -> float:
        if agent_id == "coordinator":
            return self.coordinator_value
        if agent_id == "researcher":
            return self.researcher_value
        raise KeyError(f"Unknown agent: {agent_id}")

    def fact_for(self, agent_id: str) -> str:
        if agent_id == "coordinator":
            return self.coordinator_fact
        if agent_id == "researcher":
            return self.researcher_fact
        raise KeyError(f"Unknown agent: {agent_id}")


def default_tasks() -> list[TaskSpec]:
    """Return a small fixed task set with different credit weights."""

    return [
        TaskSpec(
            task_id="release-plan",
            prompt="Prepare a safe pilot release plan.",
            coordinator_source="brief.md",
            researcher_source="evidence.md",
            coordinator_fact="the pilot runs for 7 days",
            researcher_fact="exposure is capped at 10 percent",
            coordinator_value=0.55,
            researcher_value=0.45,
        ),
        TaskSpec(
            task_id="incident-review",
            prompt="Write the follow-up plan for a service incident.",
            coordinator_source="incident.md",
            researcher_source="risk-review.md",
            coordinator_fact="rollback remains available",
            researcher_fact="latency is checked at p95",
            coordinator_value=0.45,
            researcher_value=0.55,
        ),
        TaskSpec(
            task_id="model-evaluation",
            prompt="Define a reproducible model evaluation note.",
            coordinator_source="evaluation-brief.md",
            researcher_source="evaluation-data.md",
            coordinator_fact="the evaluation uses seed 7",
            researcher_fact="calibration error is reported",
            coordinator_value=0.50,
            researcher_value=0.50,
        ),
        TaskSpec(
            task_id="data-refresh",
            prompt="Schedule a reliable data refresh.",
            coordinator_source="refresh-plan.md",
            researcher_source="schema-check.md",
            coordinator_fact="the refresh runs weekly",
            researcher_fact="the schema is validated before publishing",
            coordinator_value=0.60,
            researcher_value=0.40,
        ),
    ]


def _probe_tasks(tasks: Sequence[TaskSpec]) -> list[TaskSpec]:
    """Create disjoint task ids/facts for held-out credit evaluation."""

    return [
        replace(
            task,
            task_id=f"{task.task_id}-probe",
            prompt=f"{task.prompt} Probe instance.",
            coordinator_source=f"probe-{task.coordinator_source}",
            researcher_source=f"probe-{task.researcher_source}",
            coordinator_fact=f"{task.coordinator_fact} on the probe instance",
            researcher_fact=f"{task.researcher_fact} on the probe instance",
        )
        for task in tasks
    ]


@dataclass
class StepRecord:
    """A trace item that is sufficient for offline credit evaluation."""

    prompt_id: str
    group_id: str
    rollout_id: str
    agent_id: str
    step_id: int
    role: str
    phase: str
    action: str
    tool: str | None
    message: str | None
    observation: str
    q_i: float | None
    token_cost: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RolloutCost:
    """Token and execution counters for one rollout."""

    read_calls: int
    write_calls: int
    message_calls: int
    total_tokens: int
    wall_time_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    write_lock_wait_ms: float = 0.0

    @property
    def tool_calls(self) -> int:
        return self.read_calls + self.write_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_calls": self.read_calls,
            "write_calls": self.write_calls,
            "message_calls": self.message_calls,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_time_ms": round(self.wall_time_ms, 3),
            "write_lock_wait_ms": round(self.write_lock_wait_ms, 3),
        }


@dataclass
class RolloutResult:
    """One environment trajectory and its outcome/credit metadata."""

    task_id: str
    method: str
    group_id: str
    rollout_id: str
    seed: int
    model: str
    agent_ids: tuple[str, ...]
    read_actions: dict[str, bool]
    q_values: dict[str, float]
    final_artifact: str
    task_success: bool
    coverage_score: float
    team_reward: float
    oracle_credit: dict[str, float]
    cost: RolloutCost
    steps: list[StepRecord]
    max_parallel_reads: int
    max_parallel_writes: int
    assigned_credit: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.task_id,
            "task_id": self.task_id,
            "method": self.method,
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "seed": self.seed,
            "model": self.model,
            "topology": TOPOLOGY,
            "verifier_version": VERIFIER_VERSION,
            "oracle_version": ORACLE_VERSION,
            "agent_ids": list(self.agent_ids),
            "read_actions": self.read_actions,
            "q_i": self.q_values,
            "final_artifact": self.final_artifact,
            "task_success": self.task_success,
            "coverage_score": round(self.coverage_score, 6),
            "team_reward": round(self.team_reward, 6),
            "oracle_credit": {
                key: round(value, 6)
                for key, value in self.oracle_credit.items()
            },
            "assigned_credit": {
                key: round(value, 6)
                for key, value in self.assigned_credit.items()
            },
            "cost": self.cost.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "max_parallel_reads": self.max_parallel_reads,
            "max_parallel_writes": self.max_parallel_writes,
        }


@dataclass
class RolloutBatch:
    """A collection of trajectories, grouped by prompt for GRPO updates."""

    rollouts: list[RolloutResult]

    def by_group(self) -> dict[str, list[RolloutResult]]:
        groups: dict[str, list[RolloutResult]] = {}
        for rollout in self.rollouts:
            groups.setdefault(rollout.group_id, []).append(rollout)
        return groups

    def to_dict(self) -> dict[str, Any]:
        return {"rollouts": [rollout.to_dict() for rollout in self.rollouts]}


class SharedWorkspace:
    """In-memory workspace with parallel reads and an exclusive write lock."""

    def __init__(
        self,
        sources: Mapping[str, str],
        read_latency_s: float = 0.002,
        write_latency_s: float = 0.001,
    ) -> None:
        self.sources = dict(sources)
        self.read_latency_s = read_latency_s
        self.write_latency_s = write_latency_s
        self.artifacts: dict[str, str] = {}
        self.trace: list[dict[str, Any]] = []
        self.max_parallel_reads = 0
        self.max_parallel_writes = 0
        self.write_lock_wait_ms = 0.0
        self._active_reads = 0
        self._active_writes = 0
        self._write_lock = asyncio.Lock()

    async def read(self, agent_id: str, source_id: str) -> str:
        if source_id not in self.sources:
            raise KeyError(f"Unknown source: {source_id}")
        self._active_reads += 1
        self.max_parallel_reads = max(self.max_parallel_reads, self._active_reads)
        self.trace.append({
            "event": "read_start",
            "agent_id": agent_id,
            "source_id": source_id,
            "time": time.perf_counter(),
        })
        try:
            await asyncio.sleep(self.read_latency_s)
            return self.sources[source_id]
        finally:
            self.trace.append({
                "event": "read_end",
                "agent_id": agent_id,
                "source_id": source_id,
                "time": time.perf_counter(),
            })
            self._active_reads -= 1

    async def write(self, agent_id: str, artifact_id: str, content: str) -> None:
        wait_started = time.perf_counter()
        async with self._write_lock:
            self.write_lock_wait_ms += (time.perf_counter() - wait_started) * 1000.0
            self._active_writes += 1
            self.max_parallel_writes = max(
                self.max_parallel_writes,
                self._active_writes,
            )
            self.trace.append({
                "event": "write_start",
                "agent_id": agent_id,
                "artifact_id": artifact_id,
                "time": time.perf_counter(),
            })
            try:
                await asyncio.sleep(self.write_latency_s)
                self.artifacts[artifact_id] = content
            finally:
                self.trace.append({
                    "event": "write_end",
                    "agent_id": agent_id,
                    "artifact_id": artifact_id,
                    "time": time.perf_counter(),
                })
                self._active_writes -= 1

    def snapshot(self) -> dict[str, str]:
        return dict(self.artifacts)


@dataclass(frozen=True)
class ReadDecision:
    should_read: bool
    q_i: float


@dataclass
class ReadPolicy:
    """A tiny stochastic policy whose logits can be updated offline."""

    probabilities: dict[str, float]
    min_probability: float = 0.05
    max_probability: float = 0.95

    @classmethod
    def symmetric(cls, probability: float) -> "ReadPolicy":
        return cls({agent_id: probability for agent_id in AGENT_IDS})

    def decide(self, agent_id: str, rng: random.Random) -> ReadDecision:
        probability = self.probabilities[agent_id]
        should_read = rng.random() < probability
        if should_read:
            q_i = 0.90 + rng.uniform(-0.06, 0.06)
        else:
            q_i = 0.08 + rng.uniform(-0.04, 0.04)
        return ReadDecision(should_read=should_read, q_i=max(0.0, min(1.0, q_i)))

    def update(
        self,
        agent_id: str,
        should_read: bool,
        advantage: float,
        learning_rate: float,
    ) -> None:
        probability = self.probabilities[agent_id]
        action = 1.0 if should_read else 0.0
        # Bernoulli-logit policy gradient: d log pi / d logit = action - p.
        probability += learning_rate * advantage * (action - probability)
        self.probabilities[agent_id] = max(
            self.min_probability,
            min(self.max_probability, probability),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            agent_id: round(self.probabilities[agent_id], 6)
            for agent_id in AGENT_IDS
        }


def _outcome_reward(task: TaskSpec, read_actions: Mapping[str, bool]) -> float:
    reward = sum(
        task.value_for(agent_id)
        for agent_id in AGENT_IDS
        if read_actions.get(agent_id, False)
    )
    if all(read_actions.get(agent_id, False) for agent_id in AGENT_IDS):
        reward += SUCCESS_BONUS
    return reward


def _oracle_credit(
    task: TaskSpec,
    read_actions: Mapping[str, bool],
) -> dict[str, float]:
    full_reward = _outcome_reward(task, read_actions)
    credits: dict[str, float] = {}
    for agent_id in AGENT_IDS:
        ablated = dict(read_actions)
        ablated[agent_id] = False
        credits[agent_id] = full_reward - _outcome_reward(task, ablated)
    return credits


def _coverage_score(task: TaskSpec, read_actions: Mapping[str, bool]) -> float:
    return sum(
        task.value_for(agent_id)
        for agent_id in AGENT_IDS
        if read_actions.get(agent_id, False)
    )


def _build_final_artifact(task: TaskSpec, findings: Mapping[str, str]) -> str:
    coordinator_finding = findings.get("coordinator", "MISSING")
    researcher_finding = findings.get("researcher", "MISSING")
    return (
        f"Task: {task.prompt}\n"
        f"Coordinator finding: {coordinator_finding}\n"
        f"Researcher finding: {researcher_finding}\n"
        "Decision: proceed only when both findings are present."
    )


def _finding(task: TaskSpec, agent_id: str, read: bool) -> str:
    if read:
        return task.fact_for(agent_id)
    return "MISSING"


def verify_final_artifact(task: TaskSpec, final_artifact: str) -> bool:
    """Verify the artifact outside the agent decision path."""

    return all(
        task.fact_for(agent_id) in final_artifact
        for agent_id in AGENT_IDS
    )


def _cost_from_steps(
    steps: Sequence[StepRecord],
    started: float,
    write_lock_wait_ms: float = 0.0,
) -> RolloutCost:
    read_calls = sum(
        step.phase == "read" and step.action == "read"
        for step in steps
    )
    write_calls = sum(step.phase == "write" for step in steps)
    message_calls = sum(step.phase == "communicate" for step in steps)
    input_tokens = 0
    output_tokens = 0
    for step in steps:
        if step.phase == "read":
            if step.action == "read":
                input_tokens += READ_INPUT_TOKENS
                output_tokens += READ_OUTPUT_TOKENS
            else:
                input_tokens += SKIP_INPUT_TOKENS
        elif step.phase == "communicate":
            input_tokens += MESSAGE_INPUT_TOKENS
            output_tokens += MESSAGE_OUTPUT_TOKENS
        elif step.phase == "write":
            if step.action == "persist_final":
                input_tokens += FINAL_WRITE_INPUT_TOKENS
                output_tokens += FINAL_WRITE_OUTPUT_TOKENS
            else:
                input_tokens += FINDING_WRITE_INPUT_TOKENS
                output_tokens += FINDING_WRITE_OUTPUT_TOKENS
    return RolloutCost(
        read_calls=read_calls,
        write_calls=write_calls,
        message_calls=message_calls,
        total_tokens=sum(step.token_cost for step in steps),
        wall_time_ms=(time.perf_counter() - started) * 1000.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        write_lock_wait_ms=write_lock_wait_ms,
    )


async def run_two_agent_rollout(
    task: TaskSpec,
    policy: ReadPolicy,
    *,
    method: str,
    group_id: str,
    rollout_id: str,
    seed: int,
    read_latency_s: float = 0.002,
    write_latency_s: float = 0.001,
) -> RolloutResult:
    """Execute coordinator/researcher reads in parallel and writes serially."""

    started = time.perf_counter()
    workspace = SharedWorkspace(
        task.sources(),
        read_latency_s=read_latency_s,
        write_latency_s=write_latency_s,
    )
    rng = random.Random(seed)
    decisions = {
        agent_id: policy.decide(agent_id, rng)
        for agent_id in AGENT_IDS
    }

    async def read_for_agent(agent_id: str) -> tuple[str, str]:
        source_id = (
            task.coordinator_source
            if agent_id == "coordinator"
            else task.researcher_source
        )
        decision = decisions[agent_id]
        if not decision.should_read:
            return agent_id, "No source was read."
        return agent_id, await workspace.read(agent_id, source_id)

    read_results = dict(await asyncio.gather(
        *(read_for_agent(agent_id) for agent_id in AGENT_IDS)
    ))
    read_actions = {
        agent_id: decisions[agent_id].should_read
        for agent_id in AGENT_IDS
    }
    q_values = {
        agent_id: decisions[agent_id].q_i
        for agent_id in AGENT_IDS
    }

    steps: list[StepRecord] = []
    for step_id, agent_id in enumerate(AGENT_IDS):
        decision = decisions[agent_id]
        steps.append(StepRecord(
            prompt_id=task.task_id,
            group_id=group_id,
            rollout_id=rollout_id,
            agent_id=agent_id,
            step_id=step_id,
            role=agent_id,
            phase="read",
            action="read" if decision.should_read else "skip_read",
            tool="read_source",
            message=None,
            observation=read_results[agent_id],
            q_i=decision.q_i,
            token_cost=READ_TOKEN_COST if decision.should_read else SKIP_TOKEN_COST,
        ))

    research_message = _finding(
        task,
        "researcher",
        read_actions["researcher"],
    )
    steps.append(StepRecord(
        prompt_id=task.task_id,
        group_id=group_id,
        rollout_id=rollout_id,
        agent_id="researcher",
        step_id=len(steps),
        role="researcher",
        phase="communicate",
        action="send_finding",
        tool=None,
        message=research_message,
        observation="Finding delivered to coordinator.",
        q_i=None,
        token_cost=MESSAGE_TOKEN_COST,
    ))

    findings = {
        agent_id: _finding(task, agent_id, read_actions[agent_id])
        for agent_id in AGENT_IDS
    }

    async def persist_finding(agent_id: str) -> None:
        await workspace.write(
            agent_id,
            f"finding:{agent_id}",
            findings[agent_id],
        )

    # The two writes are submitted together; SharedWorkspace serializes them.
    await asyncio.gather(*(persist_finding(agent_id) for agent_id in AGENT_IDS))
    for agent_id in AGENT_IDS:
        steps.append(StepRecord(
            prompt_id=task.task_id,
            group_id=group_id,
            rollout_id=rollout_id,
            agent_id=agent_id,
            step_id=len(steps),
            role=agent_id,
            phase="write",
            action="persist_finding",
            tool="write_artifact",
            message=None,
            observation=findings[agent_id],
            q_i=None,
            token_cost=FINDING_WRITE_TOKEN_COST,
        ))

    persisted_findings = {
        agent_id: workspace.snapshot().get(f"finding:{agent_id}", "MISSING")
        for agent_id in AGENT_IDS
    }
    final_artifact = _build_final_artifact(task, persisted_findings)
    await workspace.write("coordinator", "final", final_artifact)
    steps.append(StepRecord(
        prompt_id=task.task_id,
        group_id=group_id,
        rollout_id=rollout_id,
        agent_id="coordinator",
        step_id=len(steps),
        role="coordinator",
        phase="write",
        action="persist_final",
        tool="write_artifact",
        message=None,
        observation=final_artifact,
        q_i=None,
        token_cost=FINAL_WRITE_TOKEN_COST,
    ))

    task_success = verify_final_artifact(task, final_artifact)
    return RolloutResult(
        task_id=task.task_id,
        method=method,
        group_id=group_id,
        rollout_id=rollout_id,
        seed=seed,
        model=MODEL_NAME,
        agent_ids=AGENT_IDS,
        read_actions=read_actions,
        q_values=q_values,
        final_artifact=final_artifact,
        task_success=task_success,
        coverage_score=_coverage_score(task, read_actions),
        team_reward=_outcome_reward(task, read_actions),
        oracle_credit=_oracle_credit(task, read_actions),
        cost=_cost_from_steps(steps, started, workspace.write_lock_wait_ms),
        steps=steps,
        max_parallel_reads=workspace.max_parallel_reads,
        max_parallel_writes=workspace.max_parallel_writes,
    )


async def run_single_agent_rollout(
    task: TaskSpec,
    *,
    read_probability: float,
    group_id: str,
    rollout_id: str,
    seed: int,
    read_latency_s: float = 0.002,
    write_latency_s: float = 0.001,
) -> RolloutResult:
    """Run the one-agent sequential read/write baseline."""

    started = time.perf_counter()
    workspace = SharedWorkspace(
        task.sources(),
        read_latency_s=read_latency_s,
        write_latency_s=write_latency_s,
    )
    rng = random.Random(seed)
    read_actions: dict[str, bool] = {}
    q_values: dict[str, float] = {}
    findings: dict[str, str] = {}
    steps: list[StepRecord] = []

    for agent_id, source_id in (
        ("coordinator", task.coordinator_source),
        ("researcher", task.researcher_source),
    ):
        should_read = rng.random() < read_probability
        q_i = 0.90 if should_read else 0.08
        read_actions[agent_id] = should_read
        q_values[agent_id] = q_i
        if should_read:
            observation = await workspace.read("single-agent", source_id)
        else:
            observation = "No source was read."
        findings[agent_id] = _finding(task, agent_id, should_read)
        steps.append(StepRecord(
            prompt_id=task.task_id,
            group_id=group_id,
            rollout_id=rollout_id,
            agent_id="single-agent",
            step_id=len(steps),
            role="single-agent",
            phase="read",
            action="read" if should_read else "skip_read",
            tool="read_source",
            message=None,
            observation=observation,
            q_i=q_i,
            token_cost=READ_TOKEN_COST if should_read else SKIP_TOKEN_COST,
        ))

    final_artifact = _build_final_artifact(task, findings)
    await workspace.write("single-agent", "final", final_artifact)
    steps.append(StepRecord(
        prompt_id=task.task_id,
        group_id=group_id,
        rollout_id=rollout_id,
        agent_id="single-agent",
        step_id=len(steps),
        role="single-agent",
        phase="write",
        action="persist_final",
        tool="write_artifact",
        message=None,
        observation=final_artifact,
        q_i=None,
        token_cost=FINAL_WRITE_TOKEN_COST,
    ))

    task_success = verify_final_artifact(task, final_artifact)
    team_reward = _outcome_reward(task, read_actions)
    return RolloutResult(
        task_id=task.task_id,
        method="single-agent",
        group_id=group_id,
        rollout_id=rollout_id,
        seed=seed,
        model=MODEL_NAME,
        agent_ids=("single-agent",),
        read_actions=read_actions,
        q_values=q_values,
        final_artifact=final_artifact,
        task_success=task_success,
        coverage_score=_coverage_score(task, read_actions),
        team_reward=team_reward,
        oracle_credit={"single-agent": team_reward},
        cost=_cost_from_steps(steps, started, workspace.write_lock_wait_ms),
        steps=steps,
        max_parallel_reads=workspace.max_parallel_reads,
        max_parallel_writes=workspace.max_parallel_writes,
    )


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense system with partial pivoting and no dependencies."""

    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Singular ridge system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


@dataclass(frozen=True)
class CreditFit:
    """Ridge fit used by the minimal observable-q CAD-GRPO proxy."""

    intercept: float
    coefficients: dict[str, float]
    q_means: dict[str, float]
    r_squared: float
    valid: bool
    fallback_reason: str = ""

    def predict(self, q_values: Mapping[str, float]) -> dict[str, float]:
        return {
            agent_id: self.coefficients.get(agent_id, 0.0)
            * (
                q_values.get(agent_id, 0.0)
                - self.q_means.get(agent_id, 0.0)
            )
            for agent_id in AGENT_IDS
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept": round(self.intercept, 6),
            "coefficients": {
                key: round(value, 6)
                for key, value in self.coefficients.items()
            },
            "q_means": {
                key: round(value, 6)
                for key, value in self.q_means.items()
            },
            "r_squared": round(self.r_squared, 6),
            "valid": self.valid,
            "fallback_reason": self.fallback_reason,
        }


def fit_credit_model(
    rollouts: Sequence[RolloutResult],
    *,
    ridge_alpha: float = 1.0,
    r_squared_threshold: float = 0.1,
) -> CreditFit:
    """Fit centered team reward from [1, q_coordinator, q_researcher]."""

    invalid = CreditFit(
        intercept=0.0,
        coefficients={agent_id: 0.0 for agent_id in AGENT_IDS},
        q_means={agent_id: 0.0 for agent_id in AGENT_IDS},
        r_squared=0.0,
        valid=False,
        fallback_reason="insufficient_variation",
    )
    if len(rollouts) < 3:
        return invalid

    q_means = {
        agent_id: sum(rollout.q_values[agent_id] for rollout in rollouts)
        / len(rollouts)
        for agent_id in AGENT_IDS
    }
    features = [
        [1.0] + [
            rollout.q_values[agent_id] - q_means[agent_id]
            for agent_id in AGENT_IDS
        ]
        for rollout in rollouts
    ]
    targets = [rollout.team_reward for rollout in rollouts]
    target_mean = sum(targets) / len(targets)
    total_variance = sum((target - target_mean) ** 2 for target in targets)
    if total_variance < 1e-12:
        return invalid

    width = len(features[0])
    gram = [[0.0 for _ in range(width)] for _ in range(width)]
    rhs = [0.0 for _ in range(width)]
    for row, target in zip(features, targets):
        for left in range(width):
            rhs[left] += row[left] * target
            for right in range(width):
                gram[left][right] += row[left] * row[right]
    for index in range(1, width):
        gram[index][index] += ridge_alpha

    try:
        coefficients = _solve_linear_system(gram, rhs)
    except ValueError:
        return CreditFit(
            **{**asdict(invalid), "fallback_reason": "singular_ridge_system"}
        )

    predictions = [
        sum(coefficient * value for coefficient, value in zip(coefficients, row))
        for row in features
    ]
    residual_sum = sum(
        (target - prediction) ** 2
        for target, prediction in zip(targets, predictions)
    )
    r_squared = 1.0 - residual_sum / total_variance
    valid = math.isfinite(r_squared) and r_squared >= r_squared_threshold
    return CreditFit(
        intercept=coefficients[0],
        coefficients={
            agent_id: coefficients[index + 1]
            for index, agent_id in enumerate(AGENT_IDS)
        },
        q_means=q_means,
        r_squared=r_squared,
        valid=valid,
        fallback_reason="" if valid else "low_r_squared",
    )


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    standard_deviation = math.sqrt(variance)
    if standard_deviation < 1e-12:
        return [0.0 for _ in values]
    return [(value - mean) / standard_deviation for value in values]


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    )
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale < 1e-12 or y_scale < 1e-12:
        return None
    return numerator / (x_scale * y_scale)


def _shared_credits(rollouts: Sequence[RolloutResult]) -> list[dict[str, float]]:
    advantages = _normalize([rollout.team_reward for rollout in rollouts])
    return [
        {agent_id: advantage for agent_id in AGENT_IDS}
        for advantage in advantages
    ]


def _cad_credits(
    rollouts: Sequence[RolloutResult],
    fit: CreditFit,
) -> list[dict[str, float]]:
    if not fit.valid:
        return _shared_credits(rollouts)
    raw = [fit.predict(rollout.q_values) for rollout in rollouts]
    normalized_by_agent = {
        agent_id: _normalize([item[agent_id] for item in raw])
        for agent_id in AGENT_IDS
    }
    return [
        {
            agent_id: normalized_by_agent[agent_id][index]
            for agent_id in AGENT_IDS
        }
        for index in range(len(rollouts))
    ]


def _attach_credits(
    rollouts: Sequence[RolloutResult],
    credits: Sequence[Mapping[str, float]],
) -> None:
    for rollout, credit in zip(rollouts, credits):
        rollout.assigned_credit = dict(credit)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    train_groups: int = 24
    group_size: int = 8
    eval_rollouts: int = 128
    initial_read_probability: float = 0.55
    single_read_probability: float = 0.80
    learning_rate: float = 0.20
    ridge_alpha: float = 1.0
    r_squared_threshold: float = 0.1
    read_latency_s: float = 0.002
    write_latency_s: float = 0.001


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_rollouts(
    rollouts: Sequence[RolloutResult],
    *,
    credit_mode: str,
) -> dict[str, Any]:
    success_rate = _mean([float(rollout.task_success) for rollout in rollouts])
    cost_tokens = _mean([rollout.cost.total_tokens for rollout in rollouts])
    input_tokens = _mean([rollout.cost.input_tokens for rollout in rollouts])
    output_tokens = _mean([rollout.cost.output_tokens for rollout in rollouts])
    tool_calls = _mean([rollout.cost.tool_calls for rollout in rollouts])
    message_calls = _mean([rollout.cost.message_calls for rollout in rollouts])
    wall_time = _mean([rollout.cost.wall_time_ms for rollout in rollouts])
    lock_wait = _mean([
        rollout.cost.write_lock_wait_ms
        for rollout in rollouts
    ])

    assigned_by_agent: dict[str, list[float]] = {
        agent_id: [] for agent_id in AGENT_IDS
    }
    oracle_by_agent: dict[str, list[float]] = {
        agent_id: [] for agent_id in AGENT_IDS
    }
    for rollout in rollouts:
        if not rollout.assigned_credit:
            continue
        for agent_id in AGENT_IDS:
            if agent_id in rollout.assigned_credit:
                assigned_by_agent[agent_id].append(
                    rollout.assigned_credit[agent_id]
                )
                oracle_by_agent[agent_id].append(
                    rollout.oracle_credit[agent_id]
                )

    pooled_correlations = {
        agent_id: _pearson(
            assigned_by_agent[agent_id],
            oracle_by_agent[agent_id],
        )
        for agent_id in AGENT_IDS
    }
    task_agent_correlations: dict[str, float | None] = {}
    for task_id in sorted({rollout.task_id for rollout in rollouts}):
        task_rollouts = [rollout for rollout in rollouts if rollout.task_id == task_id]
        for agent_id in AGENT_IDS:
            task_assigned = [
                rollout.assigned_credit[agent_id]
                for rollout in task_rollouts
                if agent_id in rollout.assigned_credit
            ]
            task_oracle = [
                rollout.oracle_credit[agent_id]
                for rollout in task_rollouts
                if agent_id in rollout.assigned_credit
            ]
            task_agent_correlations[f"{task_id}:{agent_id}"] = _pearson(
                task_assigned,
                task_oracle,
            )
    cluster_values = [
        correlation
        for correlation in task_agent_correlations.values()
        if correlation is not None
    ]
    correlation = _mean(cluster_values) if cluster_values else None
    return {
        "rollouts": len(rollouts),
        "task_success_rate": round(success_rate, 6),
        "mean_rollout_cost_tokens": round(cost_tokens, 6),
        "mean_input_tokens": round(input_tokens, 6),
        "mean_output_tokens": round(output_tokens, 6),
        "mean_tool_calls": round(tool_calls, 6),
        "mean_message_calls": round(message_calls, 6),
        "mean_wall_time_ms": round(wall_time, 6),
        "mean_write_lock_wait_ms": round(lock_wait, 6),
        "credit_oracle_correlation": (
            None if correlation is None else round(correlation, 6)
        ),
        "credit_oracle_correlation_type": (
            "not-applicable"
            if not any(assigned_by_agent.values())
            else "pearson_task_agent_macro"
        ),
        "credit_oracle_correlation_by_agent": {
            agent_id: (
                None
                if pooled_correlations[agent_id] is None
                else round(pooled_correlations[agent_id], 6)
            )
            for agent_id in AGENT_IDS
        },
        "credit_oracle_correlation_by_task_agent": {
            key: (
                None if value is None else round(value, 6)
            )
            for key, value in task_agent_correlations.items()
        },
        "credit_oracle_pairs": sum(len(values) for values in assigned_by_agent.values()),
        "credit_oracle_clusters": len(cluster_values),
        "credit_oracle_task_count": len({rollout.task_id for rollout in rollouts}),
        "oracle_rollouts": 0,
        "oracle_cost_tokens": 0,
        "credit_mode": credit_mode,
    }


def _evaluation_seeds(config: ExperimentConfig) -> list[int]:
    rng = random.Random(config.seed + 100_003)
    return [rng.randrange(1, 2**31 - 1) for _ in range(config.eval_rollouts)]


async def _train_policy(
    method: str,
    tasks: Sequence[TaskSpec],
    config: ExperimentConfig,
) -> tuple[ReadPolicy, list[RolloutResult], dict[str, CreditFit], dict[str, Any]]:
    policy = ReadPolicy.symmetric(config.initial_read_probability)
    # Common random numbers keep naive GRPO and CAD-GRPO comparable.
    rng = random.Random(config.seed + 17)
    training_rollouts: list[RolloutResult] = []
    r_squared_values: list[float] = []
    fallback_count = 0

    for group_index in range(config.train_groups):
        task = tasks[group_index % len(tasks)]
        group_id = f"{method}-group-{group_index}"
        group: list[RolloutResult] = []
        for rollout_index in range(config.group_size):
            rollout_seed = rng.randrange(1, 2**31 - 1)
            group.append(await run_two_agent_rollout(
                task,
                policy,
                method=method,
                group_id=group_id,
                rollout_id=f"{group_id}-rollout-{rollout_index}",
                seed=rollout_seed,
                read_latency_s=config.read_latency_s,
                write_latency_s=config.write_latency_s,
            ))

        if method == "naive-grpo":
            credits = _shared_credits(group)
        else:
            fit = fit_credit_model(
                group,
                ridge_alpha=config.ridge_alpha,
                r_squared_threshold=config.r_squared_threshold,
            )
            r_squared_values.append(fit.r_squared)
            if not fit.valid:
                fallback_count += 1
            credits = _cad_credits(group, fit)
        _attach_credits(group, credits)

        for rollout, credit in zip(group, credits):
            for agent_id in AGENT_IDS:
                policy.update(
                    agent_id,
                    rollout.read_actions[agent_id],
                    credit[agent_id],
                    config.learning_rate,
                )
        training_rollouts.extend(group)

    models: dict[str, CreditFit] = {}
    if method == "cad-grpo":
        # Fit once on training tasks so probe task ids remain genuinely held out.
        models["__global__"] = fit_credit_model(
            training_rollouts,
            ridge_alpha=config.ridge_alpha,
            r_squared_threshold=config.r_squared_threshold,
        )

    training_summary = _summarize_rollouts(
        training_rollouts,
        credit_mode="shared-team-advantage" if method == "naive-grpo" else "q-ridge",
    )
    training_summary["groups"] = config.train_groups
    training_summary["group_size"] = config.group_size
    training_summary["policy_after_training"] = policy.to_dict()
    training_summary["cad_group_r_squared_mean"] = round(
        _mean(r_squared_values),
        6,
    ) if r_squared_values else None
    training_summary["cad_group_fallback_rate"] = round(
        fallback_count / config.train_groups,
        6,
    ) if config.train_groups else 0.0
    if models:
        training_summary["final_credit_models"] = {
            model_id: fit.to_dict()
            for model_id, fit in models.items()
        }
    return policy, training_rollouts, models, training_summary


async def _evaluate_two_agent_policy(
    method: str,
    policy: ReadPolicy,
    tasks: Sequence[TaskSpec],
    config: ExperimentConfig,
    seeds: Sequence[int],
    models: Mapping[str, CreditFit],
) -> list[RolloutResult]:
    rollouts: list[RolloutResult] = []
    for index, seed in enumerate(seeds):
        task = tasks[index % len(tasks)]
        rollout = await run_two_agent_rollout(
            task,
            policy,
            method=method,
            group_id=f"{method}-probe",
            rollout_id=f"{method}-probe-{index}",
            seed=seed,
            read_latency_s=config.read_latency_s,
            write_latency_s=config.write_latency_s,
        )
        if method == "naive-grpo":
            rollout.assigned_credit = {
                agent_id: rollout.team_reward
                for agent_id in AGENT_IDS
            }
        else:
            fit = models.get("__global__") or models.get(task.task_id)
            if fit is not None and fit.valid:
                rollout.assigned_credit = fit.predict(rollout.q_values)
            else:
                rollout.assigned_credit = {
                    agent_id: rollout.team_reward
                    for agent_id in AGENT_IDS
                }
        rollouts.append(rollout)
    return rollouts


async def _evaluate_single_agent(
    tasks: Sequence[TaskSpec],
    config: ExperimentConfig,
    seeds: Sequence[int],
) -> list[RolloutResult]:
    rollouts: list[RolloutResult] = []
    for index, seed in enumerate(seeds):
        task = tasks[index % len(tasks)]
        rollouts.append(await run_single_agent_rollout(
            task,
            read_probability=config.single_read_probability,
            group_id="single-agent-probe",
            rollout_id=f"single-agent-probe-{index}",
            seed=seed,
            read_latency_s=config.read_latency_s,
            write_latency_s=config.write_latency_s,
        ))
    return rollouts


@dataclass
class ExperimentReport:
    """JSON-friendly result of the three-method comparison."""

    config: ExperimentConfig
    methods: dict[str, dict[str, Any]]
    task_ids: list[str]
    probe_task_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "task_ids": self.task_ids,
            "probe_task_ids": self.probe_task_ids,
            "methods": self.methods,
            "comparison_contract": {
                "two_agent_methods": ["naive-grpo", "cad-grpo"],
                "common_training_seeds": True,
                "common_probe_seeds": True,
                "common_group_size": self.config.group_size,
                "common_train_groups": self.config.train_groups,
                "single_agent_is_architecture_baseline": True,
            },
            "notes": [
                "This is a synthetic policy/data-flow smoke test, not real LLM training.",
                "q_i is an observable synthetic confidence proxy; oracle credit is leave-one-agent-out outcome marginal.",
                "single-agent credit/oracle correlation is not applicable because it has one joint decision-maker.",
            ],
        }


async def run_experiment_async(
    config: ExperimentConfig | None = None,
    tasks: Sequence[TaskSpec] | None = None,
) -> ExperimentReport:
    config = config or ExperimentConfig()
    tasks = list(tasks or default_tasks())
    if not tasks:
        raise ValueError("At least one task is required")
    if config.group_size < 2:
        raise ValueError("group_size must be at least 2")
    if config.train_groups < 0 or config.eval_rollouts < 1:
        raise ValueError("train_groups must be non-negative and eval_rollouts positive")

    probe_tasks = _probe_tasks(tasks)
    seeds = _evaluation_seeds(config)
    methods: dict[str, dict[str, Any]] = {}

    single_rollouts = await _evaluate_single_agent(probe_tasks, config, seeds)
    methods["single-agent"] = {
        "evaluation": _summarize_rollouts(
            single_rollouts,
            credit_mode="not-applicable-single-decision-maker",
        ),
        "policy": {
            "read_probability": config.single_read_probability,
        },
    }

    for method in ("naive-grpo", "cad-grpo"):
        policy, training_rollouts, models, training_summary = await _train_policy(
            method,
            tasks,
            config,
        )
        eval_rollouts = await _evaluate_two_agent_policy(
            method,
            policy,
            probe_tasks,
            config,
            seeds,
            models,
        )
        methods[method] = {
            "training": training_summary,
            "evaluation": _summarize_rollouts(
                eval_rollouts,
                credit_mode=(
                    "shared-team-reward" if method == "naive-grpo" else "q-ridge-prediction"
                ),
            ),
            "policy": policy.to_dict(),
            "training_rollouts": len(training_rollouts),
        }

    return ExperimentReport(
        config=config,
        methods=methods,
        task_ids=[task.task_id for task in tasks],
        probe_task_ids=[task.task_id for task in probe_tasks],
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    tasks: Sequence[TaskSpec] | None = None,
) -> ExperimentReport:
    """Synchronous entry point for notebooks, tests, and the CLI."""

    return asyncio.run(run_experiment_async(config=config, tasks=tasks))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal two-agent GRPO/CAD-GRPO smoke experiment."
    )
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--train-groups", type=int, default=ExperimentConfig.train_groups)
    parser.add_argument("--group-size", type=int, default=ExperimentConfig.group_size)
    parser.add_argument("--eval-rollouts", type=int, default=ExperimentConfig.eval_rollouts)
    parser.add_argument("--learning-rate", type=float, default=ExperimentConfig.learning_rate)
    parser.add_argument("--jsonl", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        train_groups=args.train_groups,
        group_size=args.group_size,
        eval_rollouts=args.eval_rollouts,
        learning_rate=args.learning_rate,
    )
    report = run_experiment(config)
    payload = report.to_dict()
    if args.jsonl:
        output_path = pathlib.Path(args.jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for method in METHODS:
                handle.write(json.dumps({
                    "method": method,
                    "evaluation": payload["methods"][method]["evaluation"],
                }) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
