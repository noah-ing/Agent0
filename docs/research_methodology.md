# Agent0 Demo Research Methodology

## Objectives
- Reproduce the Agent0 co-evolution paradigm (curriculum + executor) with tool-augmented RL on commodity hardware.
- Deliver an auditable, extensible demo that mirrors the paper's academic rigor and enables rapid iteration on reward design, tool orchestration, and evaluation.

## Guiding Principles
1. **Problem Framing**: Maintain a living positioning memo capturing hypotheses, assumptions, and success metrics (pass@1 uplift, tool-usage efficiency, curriculum diversity).
2. **Feasibility Spikes**: Run small pilot jobs (toy math tasks, 1–2 tool calls) before scaling RL loops. Log every spike with configs + seeds.
3. **Modular Infrastructure**: Config-first code (YAML), reproducible env via `uv` + Brew, W&B tracking, and deterministic data pipelines.
4. **Iterative Validation**: Unit tests for sandbox/tooling, ablations for algorithms, baseline comparisons against frozen Qwen3 with/without tool.
5. **Transparent Collaboration**: Weekly memos, structured PRs, TODO backlog, and disciplined experiment logging.
6. **Reproducibility**: Version datasets/checkpoints (DVC-ready), archive prompts, judge scripts, and hardware notes.
7. **Reflection Loops**: Post-iteration retros, adjust hypotheses, document learnings before expanding scope.

## Workflow Stages
1. **Literature + Benchmark Review**: Summaries of Agent0, R-Zero, Socratic-Zero, SPIRAL, plus current (Nov 2025) tool ecosystems.
2. **Environment Bring-Up**: Brew installs (`uv`, `sandfuzz`, `qwen-vllm`), `uv` env, GPU/remote access notes.
3. **Tooling Validation**: Deterministic SandFuzz sandbox tests, stop-go protocol harness, error capture.
4. **Curriculum Agent Prototyping**: Prompt tuning, reward signal simulation (uncertainty/tool/repetition) using synthetic executor responses.
5. **Executor Pilot**: Self-consistency voting, ADPO math, stress tests on ambiguous tasks.
6. **Full Co-Evolution Loop**: Alternating curriculum/executor training with frontier filtering, logging, and judge integration.
7. **Evaluation + Reporting**: OpenCompass harness, GPT-4o Mini verifier, pass@k metrics, difficulty drift analysis, and ablation schedule.

## Milestones
- M0: Repo scaffold + tooling smoke tests.
- M1: Curriculum/executor prompts finalized; sandbox loop demo.
- M2: Frontier filtering + ADPO training over mini dataset.
- M3: Full demo iteration with telemetry dashboards and report.
- M4: Benchmark sweep + ablations aligned with paper tables.

## Risk Mitigation
- **Judge Dependence**: Provide pluggable verifier (GPT-4o Mini vs local sympy grader) to avoid single point of failure.
- **Tool Abuse**: Hard cap tool rewards, monitor SandFuzz logs, add anti-loop heuristics.
- **Hardware Limits**: Parameterize batch sizes and rollout counts; support CPU-only debug mode.

This document serves as the Stanford-style methodology reference for all subsequent workstreams.
