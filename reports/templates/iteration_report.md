# Agent0 Iteration Report Template

> Duplicate this file into `reports/iter_XXX.md` and replace bracketed tokens.

## Run Metadata
- **Date:** {{DATE}}
- **Run Name:** {{RUN_NAME}}
- **Telemetry File:** {{TELEMETRY_PATH}}
- **Git Commit:** {{GIT_SHA}}
- **Data Sources:** {{DATASETS}}

## Curriculum Summary
- Total prompts sampled: {{CURRICULUM_BATCH}}
- Mean reward: {{MEAN_REWARD}}
- Reward breakdown plot: attach from W&B or `reports/figures/reward_breakdown.png`.
- Notable task seeds:
  1. {{TASK_1}}
  2. {{TASK_2}}

## Frontier Filtering
- Accepted / total: {{FRONTIER_ACCEPTED}} / {{FRONTIER_TOTAL}}
- Consistency band: [{{FILTER_LOW}}, {{FILTER_HIGH}}]
- Judge pass rate: {{JUDGE_PASS_RATE}}
- Common rejection reasons: {{REJECTIONS}}

## Executor Rollouts
- Mean turns per rollout: {{MEAN_TURNS}}
- Tool events per rollout (p50 / p90): {{P50_TOOLS}} / {{P90_TOOLS}}
- Example trace: link to `data/rollouts/{{TRACE_FILE}}`

## Training Signals
- GRPO reward mean/std: {{GRPO_MEAN}} / {{GRPO_STD}}
- ADPO adv mean/std: {{ADPO_MEAN}} / {{ADPO_STD}}
- TRL backend status: {{TRL_STATUS}}

## OpenCompass Snapshot (optional)
- Suite: {{EVAL_SUITE}}
- Key metrics:
  - GSM8K acc: {{GSM8K_ACC}}
  - MATH acc: {{MATH_ACC}}
  - BBH avg: {{BBH_ACC}}

## Postmortem / Next Actions
- Successes: {{WINS}}
- Issues: {{ISSUES}}
- Next iteration focus: {{NEXT_STEP}}
