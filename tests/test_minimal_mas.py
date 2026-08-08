from __future__ import annotations

import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.minimal_mas import (  # noqa: E402
    AGENT_IDS,
    ExperimentConfig,
    ReadPolicy,
    RolloutCost,
    RolloutResult,
    TaskSpec,
    _oracle_credit,
    _outcome_reward,
    _pearson,
    fit_credit_model,
    run_experiment,
    run_single_agent_rollout,
    run_two_agent_rollout,
)


TASK = TaskSpec(
    task_id="test-task",
    prompt="Prepare a test note.",
    coordinator_source="brief.md",
    researcher_source="evidence.md",
    coordinator_fact="the brief fact is present",
    researcher_fact="the evidence fact is present",
    coordinator_value=0.6,
    researcher_value=0.4,
)


def test_two_agent_rollout_reads_in_parallel_and_writes_serially() -> None:
    result = asyncio.run(run_two_agent_rollout(
        TASK,
        ReadPolicy.symmetric(1.0),
        method="cad-grpo",
        group_id="test-group",
        rollout_id="test-rollout",
        seed=11,
        read_latency_s=0.01,
        write_latency_s=0.001,
    ))

    assert result.task_success
    assert result.max_parallel_reads == 2
    assert result.max_parallel_writes == 1
    assert result.cost.read_calls == 2
    assert result.cost.write_calls == 3
    assert result.cost.message_calls == 1
    assert result.cost.input_tokens + result.cost.output_tokens == 102
    assert result.cost.write_lock_wait_ms >= 0.0
    assert TASK.coordinator_fact in result.final_artifact
    assert TASK.researcher_fact in result.final_artifact
    payload = result.to_dict()
    assert payload["topology"] == "coordinator+researcher/read-parallel-write-serial"
    assert payload["verifier_version"] == "required-facts-v1"
    assert payload["oracle_version"] == "leave-one-agent-out-total-marginal-v1"


def test_single_agent_is_sequential_and_has_no_intermediate_writes() -> None:
    result = asyncio.run(run_single_agent_rollout(
        TASK,
        read_probability=1.0,
        group_id="single-group",
        rollout_id="single-rollout",
        seed=11,
        read_latency_s=0.001,
        write_latency_s=0.001,
    ))

    assert result.task_success
    assert result.max_parallel_reads == 1
    assert result.max_parallel_writes == 1
    assert result.cost.read_calls == 2
    assert result.cost.write_calls == 1
    assert result.cost.message_calls == 0


def _synthetic_rollout(
    index: int,
    coordinator_read: bool,
    researcher_read: bool,
) -> RolloutResult:
    actions = {
        "coordinator": coordinator_read,
        "researcher": researcher_read,
    }
    return RolloutResult(
        task_id=TASK.task_id,
        method="cad-grpo",
        group_id="fit-group",
        rollout_id=f"fit-{index}",
        seed=index,
        model="test",
        agent_ids=AGENT_IDS,
        read_actions=actions,
        q_values={
            "coordinator": float(coordinator_read),
            "researcher": float(researcher_read),
        },
        final_artifact="",
        task_success=coordinator_read and researcher_read,
        coverage_score=0.0,
        team_reward=_outcome_reward(TASK, actions),
        oracle_credit=_oracle_credit(TASK, actions),
        cost=RolloutCost(0, 0, 0, 0, 0.0),
        steps=[],
        max_parallel_reads=0,
        max_parallel_writes=0,
    )


def test_ridge_credit_fit_tracks_known_oracle_marginals() -> None:
    rollouts = [
        _synthetic_rollout(0, False, False),
        _synthetic_rollout(1, True, False),
        _synthetic_rollout(2, False, True),
        _synthetic_rollout(3, True, True),
    ]

    fit = fit_credit_model(rollouts, ridge_alpha=1.0, r_squared_threshold=0.1)
    predicted = [fit.predict(rollout.q_values) for rollout in rollouts]
    predicted_values = [
        item[agent_id]
        for item in predicted
        for agent_id in AGENT_IDS
    ]
    oracle_values = [
        rollout.oracle_credit[agent_id]
        for rollout in rollouts
        for agent_id in AGENT_IDS
    ]

    assert fit.valid
    assert fit.r_squared > 0.5
    assert fit.coefficients["coordinator"] > 0.0
    assert fit.coefficients["researcher"] > 0.0
    correlation = _pearson(predicted_values, oracle_values)
    assert correlation is not None
    assert correlation > 0.8


def test_experiment_report_contains_only_requested_methods() -> None:
    report = run_experiment(ExperimentConfig(
        seed=3,
        train_groups=4,
        group_size=4,
        eval_rollouts=12,
    ))
    payload = report.to_dict()

    assert set(payload["methods"]) == {"single-agent", "naive-grpo", "cad-grpo"}
    assert set(payload["task_ids"]).isdisjoint(payload["probe_task_ids"])
    assert payload["comparison_contract"]["common_training_seeds"]
    for method in payload["methods"].values():
        evaluation = method["evaluation"]
        assert 0.0 <= evaluation["task_success_rate"] <= 1.0
        assert evaluation["mean_rollout_cost_tokens"] > 0
    assert payload["methods"]["single-agent"]["evaluation"][
        "credit_oracle_correlation"
    ] is None
    assert payload["methods"]["cad-grpo"]["evaluation"][
        "credit_oracle_pairs"
    ] > 0
