# Historical implementation plan

> Planning note, written before the prototype was evaluated. Items below are
> proposed work, not completed experiments or evidence of Agent0 reproduction.

## Research question

Can a small, auditable harness reproduce parts of Agent0's curriculum/executor
orchestration and eventually test co-training claims under controlled baselines?

The current repository does not answer that question. It contains endpoint
clients, filtering and reward prototypes, trainer interfaces, and an independent
OpenCompass evaluation of an off-the-shelf model. It does not contain a trained
checkpoint or a validated optimizer-backed co-evolution run.

## Proposed method

1. Define hypotheses and success criteria before each experiment, including a
   frozen-model baseline and a clear primary metric.
2. Validate tool execution and logging on small synthetic tasks before running
   model-generated code or paid endpoints at scale.
3. Version the exact configuration, commit, prompt templates, dataset revisions,
   seeds, checkpoint identifiers, and sanitized run artifacts.
4. Separate orchestration signals from parameter updates. A reward calculation,
   accepted frontier sample, or backend callback is not evidence of learning.
5. Compare trained and frozen baselines under the same model, tools, prompts,
   token budget, and evaluation protocol.
6. Report failed runs and uncertainty alongside successful results.

## Proposed milestones

- **M0 — Harness checks:** unit tests for filtering, rewards, telemetry, and the
  sandbox adapter.
- **M1 — Orchestration smoke test:** versioned traces from a small curriculum and
  executor loop, without a training claim.
- **M2 — Optimizer integration:** prove that configured trainer steps update a
  checkpoint and record before/after hashes.
- **M3 — Controlled experiment:** compare trained and frozen baselines with
  retained manifests and repeated runs.
- **M4 — External evaluation:** run a preregistered benchmark and publish enough
  sanitized evidence for independent review.

## Primary risks

- **Untrusted code:** run model-generated code in a disposable, least-privilege
  environment; the adapter is not itself a hardened security boundary.
- **Judge dependence:** use deterministic graders where possible and measure
  sensitivity to judge choice.
- **Hosted-model drift:** record provider model identifiers and dates, and avoid
  claiming deterministic reproduction from API-backed scores alone.
- **Missing provenance:** treat a metric without its raw manifest and artifacts as
  a historical observation, not a reproducible result.
- **Hardware constraints:** keep pilot runs small and publish the actual compute
  and wall-clock budget.
