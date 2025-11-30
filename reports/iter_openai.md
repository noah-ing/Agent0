# Agent0 Iteration Report Template

> Duplicate this file into `reports/iter_XXX.md` and replace bracketed tokens.

## Run Metadata
- **Date:** 2025-11-26
- **Run Name:** iter_openai
- **Telemetry File:** reports/telemetry_openai.jsonl
- **Git Commit:** unknown
- **Data Sources:** N/A

## Curriculum Summary
- Total prompts sampled: 2
- Mean reward: 0.4830
- Reward breakdown plot: attach from W&B or `reports/figures/reward_breakdown.png`.
- Notable task seeds:
  1. GSM8K seed
  2. MATH seed

## Frontier Filtering
- Accepted / total: 0 / 2
- Consistency band: [0.30, 0.80]
- Judge pass rate: 0.0000
- Common rejection reasons: repetition=0, out-of-band=0

## Executor Rollouts
- Mean turns per rollout: 5.33
- Tool events per rollout (p50 / p90): 2.00 / 2.50
- Example trace: link to `data/rollouts/(attach rollout file)`

## Training Signals
- GRPO reward mean/std: 0.4830 / 0.0010
- ADPO adv mean/std: N/A / N/A
- TRL backend status: not configured

## OpenCompass Snapshot (optional)
- Suite: math-lite
- Key metrics:
  - GSM8K acc: N/A
  - MATH acc: N/A
  - BBH avg: N/A

## Postmortem / Next Actions
- Successes: TBD
- Issues: TBD
- Next iteration focus: TBD
