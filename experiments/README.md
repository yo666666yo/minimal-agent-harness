# Minimal MAS experiment

This directory contains the first data-flow smoke test for a two-agent setup:

- `coordinator` reads `brief.md` and later writes the final artifact.
- `researcher` reads `evidence.md` and sends one finding to the coordinator.
- The two reads are submitted concurrently.
- Finding writes and the final write pass through an exclusive workspace lock.

The runner compares exactly three methods:

1. `single-agent`: one decision maker reads both sources sequentially and writes once.
2. `naive-grpo`: two-agent rollouts update both policies with the same normalized team reward.
3. `cad-grpo`: two-agent rollouts fit centered `team_reward ~ 1 + (q_coordinator - mean(q_coordinator)) + (q_researcher - mean(q_researcher))` with ridge regression. The per-agent `beta_i * (q_i - mean(q_i))` prediction is used as the credit signal; low-`R^2` groups fall back to the shared team advantage.

Run it from the repository root:

```powershell
python -m experiments.minimal_mas --train-groups 24 --group-size 8 --eval-rollouts 128
```

The JSON report contains `task_success_rate`, mean input/output token cost, tool/message counts, write-lock wait, wall time, policy state, CAD fit diagnostics, and held-out `credit_oracle_correlation`. Training uses the base task ids; evaluation uses disjoint `*-probe` task ids and a global CAD fit learned only from training rollouts. The primary correlation is a macro average of Pearson correlations within each `task_id x agent_id` cluster; `credit_oracle_clusters` reports how many clusters had enough variation. The oracle is a leave-one-agent-out total marginal of the synthetic outcome reward, not an intervention on a real model. `q_i` is an observable synthetic confidence proxy, so this first result validates plumbing and estimator behavior only.

`single-agent` is a frozen one-decision-maker architecture baseline. `naive-grpo` and `cad-grpo` share the two-agent topology, task schedule, probe seeds, rollout budget, and policy-update budget; their intended direct comparison is the credit rule. The single-agent comparison also changes topology and training status, so it is not a causal estimate of the CAD-GRPO gain.

For a compact JSONL summary:

```powershell
python -m experiments.minimal_mas --jsonl artifacts\minimal_mas_summary.jsonl
```
