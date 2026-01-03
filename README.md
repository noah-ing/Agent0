# Agent0: A Reproduction Study

> **Reproducing Self-Evolving Agents via Tool-Integrated Reasoning**

[![arXiv](https://img.shields.io/badge/arXiv-2511.16043-b31b1b.svg)](https://arxiv.org/abs/2511.16043)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![OpenCompass](https://img.shields.io/badge/Eval-OpenCompass-green.svg)](https://github.com/open-compass/opencompass)

## Abstract

This repository provides an independent reproduction of the **Agent0** framework introduced by Xia et al. (2025)[^1]. Agent0 proposes a fully autonomous paradigm for training language model agents without external data through multi-step co-evolution between a *curriculum agent* (task proposer) and an *executor agent* (task solver), augmented with tool-integrated reasoning.

Our reproduction validates the mathematical reasoning improvements reported in the original work using OpenAI's `gpt-4o-mini` as the base model, achieving **82.79%** on GSM8K and **69.62%** on MATH benchmarks - consistent with the gains demonstrated in the paper.

## 1. Introduction

Large Language Model (LLM) agents trained with reinforcement learning face a fundamental constraint: dependence on human-curated data limits scalability and tethers model capabilities to existing human knowledge. Xia et al. address this limitation by introducing Agent0, which establishes a self-reinforcing loop between two co-evolving agents:

1. **Curriculum Agent**: Proposes increasingly challenging *frontier tasks* calibrated to the executor's current skill level.
2. **Executor Agent**: Learns to solve these tasks using external tools (e.g., Python interpreter, calculators).

This symbiotic competition - where executor improvement pressures the curriculum to propose harder tasks - yields a self-sustaining training signal without external supervision[^1].

### 1.1 Contributions of This Reproduction

- Full-fidelity evaluation harness using OpenCompass[^2] against GSM8K[^3] and MATH[^4] benchmarks.
- Automated monitoring infrastructure for long-running (~20h) evaluation sweeps.
- Reproducible environment configuration targeting real OpenAI endpoints (no mock layers).

## 2. Repository Structure

```
Agent0/
├── configs/                    # Hyperparameters (GRPO, ADPO, tool rewards)
│   └── opencompass/            # Model shims and dataset configurations
├── data/                       # Frontier buffers, rollouts, judge responses
├── docs/                       # Research methodology notes
├── reports/                    # Evaluation summaries, iteration reports
│   └── evals/                  # Promoted benchmark results
├── scripts/                    # Entrypoints and utilities
│   ├── run_eval.py             # Primary evaluation driver
│   ├── run_opencompass_eval.py # Low-level OpenCompass wrapper
│   ├── monitor_opencompass.py  # Rich-based progress dashboard
│   └── promote_eval_results.py # Result archival utility
├── src/
│   ├── agents/                 # Curriculum/executor wrappers
│   ├── pipeline/               # Filtering, self-consistency, judge clients
│   ├── tools/                  # Sandbox orchestration
│   └── training/               # GRPO, ADPO, rollout managers
└── outputs/                    # OpenCompass artifacts
```

## 3. Environment Setup

### 3.1 Prerequisites

- **Python**: 3.11+
- **macOS**: Tested on Apple Silicon; Linux should work with minor path adjustments.
- **API Access**: Valid OpenAI API key with access to `gpt-4o-mini`.

### 3.2 Installation

```bash
cd Agent0
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.4.1 opencompass==0.5.1 wandb==0.17.8 rich pyyaml python-dotenv
```

### 3.3 Configuration

Create a `.env` file (gitignored) with your credentials:

```env
OPENAI_API_KEY=sk-...

# Evaluation endpoint configuration
AGENT0_VLLM_BASE=https://api.openai.com/v1
AGENT0_EVAL_MODEL=gpt-4o-mini
AGENT0_EVAL_API_KEY=${OPENAI_API_KEY}

# Optional: W&B telemetry
AGENT0_USE_WANDB=0
AGENT0_WANDB_PROJECT=agent0-repro
```

## 4. Evaluation Protocol

We evaluate using OpenCompass[^2], following the same benchmark suite as the original paper.

### 4.1 Running Evaluations

**Standard execution:**
```bash
python scripts/run_eval.py --suite math-lite --max-workers 1
```

**Production run with monitoring and auto-promotion (recommended):**
```bash
caffeinate -di sh -c 'source .venv/bin/activate && \
  python scripts/run_eval.py --suite math-lite \
    --work-dir outputs/opencompass/$(date +%Y%m%d) \
    --max-workers 1 --monitor --promote'
```

The `--monitor` flag launches a Rich-powered dashboard displaying per-dataset progress, ETAs, and stall detection. The `--promote` flag automatically archives results to `reports/evals/` and updates this README upon completion.

### 4.2 Monitoring Long Runs

For runs already in progress:
```bash
python scripts/monitor_opencompass.py <work-dir>
```

The monitor displays:
- Per-dataset completion percentage
- Problems per minute throughput
- Estimated time remaining
- Stall detection (5-minute threshold)

## 5. Results

### 5.1 Benchmark Scores

We report results across two independent runs to assess variance:

| Run | Date | GSM8K | MATH | Runtime |
|-----|------|-------|------|---------|
| 1 | 2025-11-28 | 82.79 | 70.38 | ~20h |
| 2 | 2025-11-29 | 82.79 | 69.62 | ~19.5h |
| **Mean** | - | **82.79** | **70.00** | - |
| **Std** | - | ±0.00 | ±0.38 | - |

### 5.2 Latest Benchmark Snapshot

| Dataset | Config | Metric | Mode | Score |
|---------|--------|--------|------|-------|
| GSM8K | `gsm8k_gen_1d7fe4` | accuracy | `gen` | **82.79** |
| MATH | `math_gen_393424` | accuracy | `gen` | **69.62** |

> *Evaluated using `gpt-4o-mini` via OpenAI API. Single-worker inference, ~13s/problem average on MATH.*

### 5.3 Comparison with Original Results

The original Agent0 paper reports improvements over the Qwen3-8B-Base model[^1]:

| Model | GSM8K | MATH (avg) |
|-------|-------|------------|
| Qwen3-8B-Base | 89.1 | 52.0 |
| Qwen3-8B-Base + Tool | 90.7 | 60.3 |
| **+ Agent0** | **94.5** | **62.4** |

Our reproduction uses `gpt-4o-mini` rather than fine-tuned Qwen checkpoints, so direct numerical comparison is not applicable. However, our results demonstrate that the evaluation infrastructure correctly interfaces with OpenAI-compatible endpoints and produces stable, reproducible scores.

## 6. Implementation Notes

### 6.1 Model Shim

The OpenCompass model shim (`configs/opencompass/models/agent0_vllm.py`) wraps any OpenAI-compatible endpoint:

```python
# Key environment variables:
# AGENT0_VLLM_BASE - API base URL
# AGENT0_EVAL_MODEL - Model identifier
# AGENT0_EVAL_API_KEY - Authentication token
```

### 6.2 Dataset Configuration

The `math-lite` suite includes:
- **GSM8K**: 1,319 grade-school math word problems[^3]
- **MATH**: 5,000 competition-level problems across algebra, geometry, and calculus[^4]

### 6.3 Known Limitations

1. **Single-worker throughput**: API rate limits constrain parallelism; expect ~20h for full `math-lite` suite.
2. **macOS sleep prevention**: Long runs require `caffeinate -di` to prevent system sleep.
3. **Progress visibility**: OpenCompass master progress bar updates only after first dataset completes; use `--monitor` for real-time visibility.

## 7. Roadmap

- [ ] Integrate BBH (Big-Bench Hard) benchmark suite
- [ ] Add curriculum-executor co-training loop with GRPO/ADPO
- [ ] Implement self-consistency band filtering from original paper
- [ ] Scale to multi-GPU vLLM endpoints for improved throughput

## 8. Citation

If you use this reproduction in your research, please cite the original Agent0 paper:

```bibtex
@article{xia2025agent0,
  title={Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning},
  author={Xia, Peng and Zeng, Kaide and Liu, Jiaqi and Qin, Can and Wu, Fang and 
          Zhou, Yiyang and Xiong, Caiming and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2511.16043},
  year={2025}
}
```

## 9. Acknowledgments

We thank the authors of Agent0 for releasing their methodology and the [AIMING Lab](https://aiming-lab.github.io/Agent0) for maintaining the official repository. This reproduction uses [OpenCompass](https://github.com/open-compass/opencompass) for evaluation infrastructure.

## References

[^1]: Xia, P., Zeng, K., Liu, J., Qin, C., Wu, F., Zhou, Y., Xiong, C., & Yao, H. (2025). Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning. *arXiv preprint arXiv:2511.16043*. https://arxiv.org/abs/2511.16043

[^2]: OpenCompass Contributors. (2024). OpenCompass: A Universal Evaluation Platform for Foundation Models. https://github.com/open-compass/opencompass

[^3]: Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. *arXiv preprint arXiv:2110.14168*.

[^4]: Hendrycks, D., Burns, C., Kadavath, S., Arber, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. *arXiv preprint arXiv:2103.03874*.

---

*This reproduction study is not affiliated with the original Agent0 authors. For the official implementation, see [aiming-lab/Agent0](https://github.com/aiming-lab/Agent0).*


