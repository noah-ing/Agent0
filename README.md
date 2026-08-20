# Agent0 Evaluation Harness

> OpenCompass benchmark tooling for evaluating `gpt-4o-mini` on GSM8K and MATH.

[![Agent0 paper](https://img.shields.io/badge/arXiv-2511.16043-b31b1b.svg)](https://arxiv.org/abs/2511.16043)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![OpenCompass](https://img.shields.io/badge/Eval-OpenCompass-green.svg)](https://github.com/open-compass/opencompass)

> [!IMPORTANT]
> This repository does **not** reproduce Agent0's curriculum/executor co-training. Its recorded scores are evaluations of an off-the-shelf `gpt-4o-mini` endpoint, not gains produced by the Agent0 training method.

## Scope

This project is an independent evaluation harness inspired by the
[Agent0 paper](https://arxiv.org/abs/2511.16043). It provides:

- OpenCompass dataset configurations for GSM8K and MATH;
- an OpenAI-compatible model shim configured through environment variables;
- scripts for launching, monitoring, and promoting long-running benchmark results; and
- compact, versioned summaries from two `gpt-4o-mini` evaluation runs.

The repository also contains experimental curriculum/executor orchestration,
frontier-filtering, reward-shaping, and trainer interfaces. Those modules are
prototype scaffolding. They have not demonstrated end-to-end co-training,
produced a trained Agent0 checkpoint, or reproduced the paper's reported
training improvements. The current FlexRL backend logs batch sizes rather than
performing optimizer updates, and optional TRL bridges require separately
configured local models.

For the authors' implementation of the training method, see
[aiming-lab/Agent0](https://github.com/aiming-lab/Agent0).

## What the benchmark establishes

The checked-in summaries show that the harness can send GSM8K and MATH prompts
through OpenCompass to an OpenAI-compatible `gpt-4o-mini` endpoint and collect
the corresponding accuracy metrics.

They do **not** establish:

- curriculum-agent or executor-agent parameter updates;
- multi-step co-evolution between those agents;
- a causal improvement over a frozen-model baseline;
- parity with the paper's Qwen-based training setup; or
- independent verification of the historical runs from raw artifacts.

The full OpenCompass work directories are gitignored; this repository retains
the configurations and concise result summaries.

## Repository layout

```text
Agent0/
├── configs/
│   └── opencompass/            # Model and GSM8K/MATH dataset configs
├── docs/                       # Research planning notes
├── reports/
│   └── evals/                  # Checked-in benchmark summaries
├── scripts/
│   ├── run_eval.py             # Evaluation entry point
│   ├── run_opencompass_eval.py # OpenCompass command builder
│   ├── monitor_opencompass.py  # Progress dashboard
│   └── promote_eval_results.py # Summary promotion utility
└── src/
    ├── agents/                 # Experimental agent clients
    ├── pipeline/               # Filtering, judging, and rewards
    ├── tools/                  # Sandbox integration
    └── training/               # Experimental trainer scaffolding
```

## Setup

### Prerequisites

- Python 3.11+
- an OpenAI API key with access to `gpt-4o-mini`
- macOS or Linux (the recorded long runs used Apple Silicon macOS)

Create an environment and install the evaluation dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencompass==0.5.1 rich pyyaml python-dotenv
```

Create a gitignored `.env` file:

```dotenv
OPENAI_API_KEY=replace-with-your-key
AGENT0_VLLM_BASE=https://api.openai.com/v1
AGENT0_EVAL_MODEL=gpt-4o-mini
AGENT0_EVAL_API_KEY=replace-with-your-key
```

Never commit real credentials. OpenCompass evaluation calls can incur API
charges; review the selected datasets and endpoint before starting a run.

## Evaluation Harness (OpenCompass)

Inspect the generated command without making API calls:

```bash
python scripts/run_eval.py --suite math-lite --dry-run
```

Run the GSM8K and MATH suite:

```bash
python scripts/run_eval.py --suite math-lite --max-workers 1
```

Use the built-in monitor and promote a completed summary:

```bash
python scripts/run_eval.py \
  --suite math-lite \
  --work-dir outputs/opencompass \
  --max-workers 1 \
  --monitor \
  --promote
```

For an existing work directory:

```bash
python scripts/monitor_opencompass.py outputs/opencompass
```

## Recorded results

| Run | Date | Endpoint model | GSM8K | MATH | Report |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 2025-11-28 | `gpt-4o-mini` | 82.79 | 70.38 | [summary](reports/evals/20251128_mathlite.md) |
| 2 | 2025-11-29 | `gpt-4o-mini` | 82.79 | 69.62 | [summary](reports/evals/20251129_151443.md) |

### Latest Benchmark Snapshot

| Dataset | Config | Metric | Mode | Score |
| --- | --- | --- | --- | ---: |
| GSM8K | `gsm8k_gen_1d7fe4` | accuracy | `gen` | **82.79** |
| MATH | `math_0shot_gen_393424` | accuracy | `gen` | **69.62** |

These are historical endpoint-evaluation results. A direct numerical comparison
with the paper is not valid because the model, training state, and evaluation
protocol differ. API-backed results can also vary across provider revisions
even when temperature is set to zero.

## Experimental co-evolution code

`scripts/run_demo.py` wires together prototype curriculum and executor
clients, filtering, reward calculations, sandbox calls, and trainer adapters.
By default, the backend in `src/training/backends.py` records batch sizes
rather than performing a real optimizer update. Optional TRL adapters are
activated only when separately supplied model settings are present.

Treat this path as exploratory code, not as evidence that the Agent0 algorithm
has been reproduced.

## Security and data handling

- Keep endpoint credentials in the gitignored `.env` file and scope them to the
  minimum models and spend required for a run.
- Model-generated Python is delegated to the external SandFuzz executable. This
  repository does not treat that boundary as hardened isolation; run it in a
  disposable environment without sensitive files or unrestricted network access.
- The sandbox child process receives only a small allowlist of non-secret
  environment variables. Tool code, stdout, stderr, prompts, and rollout data can
  still be sensitive and are written beneath gitignored local artifact paths.
- Review generated artifacts before sharing them. The configured telemetry
  labels are documentation, not a general-purpose content redaction system.

## Known limitations

1. There is no end-to-end, validated curriculum/executor co-training run.
2. No trained checkpoint or controlled before/after comparison is published.
3. Raw OpenCompass output directories are not included in version control.
4. The environment is documented but not lockfile-pinned.
5. The recorded runs use a hosted model that can change independently of this repository.

## Roadmap

- [ ] Implement and test real curriculum/executor optimizer updates.
- [ ] Add unit and integration tests for the project-owned modules.
- [ ] Publish sanitized run manifests with model, prompt, dependency, and commit metadata.
- [ ] Add frozen-model baselines and controlled ablations before making training claims.

## Citation

If this harness informs research on Agent0, cite the original paper:

```bibtex
@article{xia2025agent0,
  title={Agent0: Unleashing Self-Evolving Agents from Zero Data via Tool-Integrated Reasoning},
  author={Xia, Peng and Zeng, Kaide and Liu, Jiaqi and Qin, Can and Wu, Fang and
          Zhou, Yiyang and Xiong, Caiming and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2511.16043},
  year={2025}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

This repository is not affiliated with the Agent0 authors. It uses
[OpenCompass](https://github.com/open-compass/opencompass) for evaluation and
credits the [Agent0 authors](https://aiming-lab.github.io/Agent0) for the
research direction.
