# Prototype Orchestration Smoke Test

> Historical, partial log from 2025-11-26. This was not a completed
> curriculum/executor co-training run. Raw telemetry and rollout artifacts are
> not versioned, so the retained summary values are not independently
> reproducible.

## Run metadata

- **Date:** 2025-11-26
- **Run name:** `iter_openai`
- **Git commit:** Not recorded
- **Source artifacts:** Not retained in version control

## Curriculum summary

- Prompts sampled: 2
- Reported mean reward: 0.4830
- Reward breakdown figure: Not captured
- Task seed labels: GSM8K and MATH

The reward is a computed orchestration signal; it is not evidence of a model
parameter update.

## Frontier filtering

- Accepted / total: 0 / 2
- Consistency band: [0.30, 0.80]
- Judge pass rate: 0.0000
- Detailed rejection reasons: Not retained

## Executor rollouts

- Reported mean turns per rollout: 5.33
- Reported tool events per rollout (p50 / p90): 2.00 / 2.50
- Example trace: Not retained

## Training signals

- Reported GRPO reward mean / standard deviation: 0.4830 / 0.0010
- ADPO advantage: Not produced because the frontier was empty
- TRL backend: Not configured
- Optimizer update: Not demonstrated

## OpenCompass evaluation

OpenCompass evaluation was not part of this smoke test. See the separate
[2025-11-28](evals/20251128_mathlite.md) and
[2025-11-29](evals/20251129_151443.md) benchmark summaries.

## Outcome

- The prototype generated and scored two curriculum prompts.
- No samples entered the executor-training frontier.
- No configured training backend or retained artifacts establish a model update.
- A future training claim would require a versioned commit, sanitized run
  manifest, retained traces, checkpoint identifiers, and controlled baselines.
