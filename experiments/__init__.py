"""Small, reproducible multi-agent experiment environments."""

import importlib

__all__ = [
    "AGENT_IDS",
    "METHODS",
    "ExperimentConfig",
    "ExperimentReport",
    "fit_credit_model",
    "RolloutBatch",
    "RolloutResult",
    "TaskSpec",
    "default_tasks",
    "run_single_agent_rollout",
    "run_two_agent_rollout",
    "run_experiment",
    "verify_final_artifact",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(".minimal_mas", __name__)
    return getattr(module, name)
