#!/usr/bin/env python3
"""Runs a toy Agent0 co-evolution iteration for sanity checks."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.settings import load_env
from src.settings.config_loader import load_yaml_config

load_env(ROOT)

from src.agents.base_agent import AgentConfig
from src.agents.curriculum_agent import CurriculumAgent
from src.agents.executor_agent import ExecutorAgent
from src.clients.vllm_client import VLLMHTTPClient
from src.pipeline.filtering import FilterConfig, FrontierFilter
from src.pipeline.judgers import GPT4MiniJudge
from src.pipeline.rewarding import CurriculumRewardModel
from src.tools.python_sandbox import PythonSandbox, SandboxConfig, StopGoConfig, StopGoController
from src.training.adpo import ADPOTrainer
from src.training.backends import FlexRLBackend, FlexRLConfig
from src.training.curriculum_loop import CoEvolutionLoop, LoopConfig
from src.training.grpo import GRPOTrainer
from src.training.trl_bridge import build_trl_grpo, build_trl_ppo
from src.utils.telemetry import TelemetryLogger


CURRICULUM_CONFIG_PATH = ROOT / "configs" / "curriculum.yaml"
EXECUTOR_CONFIG_PATH = ROOT / "configs" / "executor.yaml"
TOOLING_CONFIG_PATH = ROOT / "configs" / "tooling.yaml"
LOOP_CONFIG_PATH = ROOT / "configs" / "loop.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent0 co-evolution demo loop")
    parser.add_argument("--curriculum-config", type=Path, default=CURRICULUM_CONFIG_PATH)
    parser.add_argument("--executor-config", type=Path, default=EXECUTOR_CONFIG_PATH)
    parser.add_argument("--tooling-config", type=Path, default=TOOLING_CONFIG_PATH)
    parser.add_argument("--loop-config", type=Path, default=LOOP_CONFIG_PATH)
    parser.add_argument("--iterations", type=int, default=None, help="Override iterations for CoEvolution loop")
    parser.add_argument("--curriculum-batch", type=int, default=None, help="Override curriculum batch size")
    parser.add_argument("--executor-batch", type=int, default=None, help="Override executor batch size")
    parser.add_argument("--executor-samples", type=int, default=None, help="Override executor sample count (k) per task")
    parser.add_argument("--run-name", default=os.getenv("AGENT0_RUN_NAME"))
    return parser.parse_args()


def _load_configs(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    return {
        "curriculum": load_yaml_config(args.curriculum_config),
        "executor": load_yaml_config(args.executor_config),
        "tooling": load_yaml_config(args.tooling_config),
        "loop": load_yaml_config(args.loop_config) if args.loop_config.exists() else {},
    }


def _agent_config(block: Dict[str, Any], prompt_block: Dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        name=block.get("name", "mock-agent"),
        endpoint=block.get("endpoint", "http://localhost:8000/v1"),
        max_tokens=int(block.get("max_tokens", 512)),
        temperature=float(block.get("temperature", 0.2)),
        system_prompt=prompt_block.get("system", "You are an Agent0 role."),
        api_key_env=block.get("api_key_env"),
    )


def _grpo_coeffs(rewards_block: Dict[str, Any]) -> Dict[str, float]:
    def coeff(key: str, default: float = 0.0) -> float:
        return float(rewards_block.get(key, {}).get("lambda", default))

    return {
        "uncertainty": coeff("uncertainty", 0.6),
        "tool_usage": coeff("tool_usage", 0.3),
        "repetition": coeff("repetition_penalty", 0.2),
    }


def _sandbox_config(tooling: Dict[str, Any]) -> SandboxConfig:
    block = tooling.get("sandbox", {})
    telemetry = tooling.get("telemetry", {})
    log_path = telemetry.get("log_path")
    return SandboxConfig(
        engine=block.get("engine", "sandfuzz"),
        worker_pool_size=int(block.get("worker_pool_size", 1)),
        python_runtime=block.get("python_runtime", "python3.11"),
        execution_timeout_s=int(block.get("execution_timeout_s", 5)),
        memory_limit_mb=int(block.get("memory_limit_mb", 512)),
        artifact_dir=Path(block.get("artifact_dir", "data/tool_runs")),
        log_path=Path(log_path) if log_path else None,
    )


def _stopgo_config(tooling: Dict[str, Any]) -> StopGoConfig:
    block = tooling.get("stop_go", {})
    return StopGoConfig(
        trigger_regex=block.get("trigger_regex", r"```python\n([\s\S]*?)```"),
        max_code_blocks=int(block.get("max_code_blocks", 5)),
        capture_stdout=bool(block.get("capture_stdout", True)),
        capture_stderr=bool(block.get("capture_stderr", True)),
    )


def _frontier_config(executor_cfg: Dict[str, Any]) -> FilterConfig:
    filtering = executor_cfg.get("filtering", {})
    band = filtering.get("self_consistency_band", {})
    return FilterConfig(
        low=float(band.get("low", 0.3)),
        high=float(band.get("high", 0.8)),
        repetition_threshold=float(filtering.get("repetition_threshold", 0.8)),
        max_history=int(filtering.get("max_history", 256)),
    )


def _executor_runtime_cfg(executor_cfg: Dict[str, Any]) -> Dict[str, Any]:
    rollouts = executor_cfg.get("rollouts", {})
    final_markers = [marker for marker in [rollouts.get("stop_token"), "\\boxed{", "</final_answer>"] if marker]
    return {
        "max_turns": int(rollouts.get("max_dialog_turns", 4)),
        "continuation_prompt": rollouts.get("continuation_prompt"),
        "final_markers": final_markers or None,
    }


def _loop_params(loop_cfg: Dict[str, Any]) -> Dict[str, int]:
    block = loop_cfg.get("loop", loop_cfg)
    return {
        "curriculum_batch": int(block.get("curriculum_batch", 2)),
        "executor_batch": int(block.get("executor_batch", 2)),
        "iterations": int(block.get("iterations", 1)),
        "executor_samples": int(block.get("executor_samples", 4)),
    }


def _build_backend(logging_cfg: Dict[str, Any], default_run: str) -> FlexRLBackend:
    project = logging_cfg.get("project", "agent0-demo")
    run_name = logging_cfg.get("run_name", default_run)
    optimizer = logging_cfg.get("optimizer", "adamw")
    return FlexRLBackend(FlexRLConfig(project=project, run_name=run_name, optimizer=optimizer))


def main() -> None:
    args = _parse_args()
    configs = _load_configs(args)
    sandbox_cfg = _sandbox_config(configs["tooling"])
    sandbox_cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    sandbox = PythonSandbox(sandbox_cfg)
    stop_go = StopGoController(sandbox, _stopgo_config(configs["tooling"]))

    run_name = args.run_name or os.getenv("AGENT0_RUN_NAME", "demo-loop")
    telemetry = TelemetryLogger.from_env(run_name)

    def client_factory(agent_cfg: AgentConfig):
        if not agent_cfg.endpoint:
            raise SystemExit(f"Agent {agent_cfg.name} is missing an endpoint in its config.")
        return VLLMHTTPClient(agent_cfg.endpoint, api_key_env=agent_cfg.api_key_env)

    def maybe_build_grpo_trl():
        model_name = os.getenv("AGENT0_TRL_CURRICULUM_MODEL")
        if not model_name:
            return None
        grpo_args = {
            "learning_rate": float(os.getenv("AGENT0_TRL_CURRICULUM_LR", "1e-6")),
            "batch_size": int(os.getenv("AGENT0_TRL_CURRICULUM_BATCH", "64")),
        }
        try:
            return build_trl_grpo(model_name, grpo_args)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to init TRL GRPO trainer: {exc}")
            return None

    def maybe_build_ppo_trl():
        model_name = os.getenv("AGENT0_TRL_EXECUTOR_MODEL")
        if not model_name:
            return None
        ppo_args = {
            "learning_rate": float(os.getenv("AGENT0_TRL_EXECUTOR_LR", "8e-7")),
            "batch_size": int(os.getenv("AGENT0_TRL_EXECUTOR_BATCH", "64")),
        }
        try:
            return build_trl_ppo(model_name, ppo_args)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to init TRL PPO trainer: {exc}")
            return None

    curriculum_cfg = _agent_config(configs["curriculum"].get("model", {}), configs["curriculum"].get("prompt", {}))
    executor_cfg = _agent_config(configs["executor"].get("model", {}), configs["executor"].get("prompt", {}))

    judge = GPT4MiniJudge()
    reward_model = CurriculumRewardModel(judge, telemetry)
    curriculum_agent = CurriculumAgent(
        curriculum_cfg,
        client_factory,
    )
    executor_runtime = _executor_runtime_cfg(configs["executor"])
    executor_agent = ExecutorAgent(
        executor_cfg,
        client_factory,
        stop_go,
        max_turns=executor_runtime["max_turns"],
        continuation_prompt=executor_runtime.get("continuation_prompt"),
        final_markers=executor_runtime.get("final_markers"),
        logger=telemetry,
    )
    frontier_filter = FrontierFilter(_frontier_config(configs["executor"]), judge, logger=telemetry)

    curriculum_backend = _build_backend(configs["curriculum"].get("logging", {}), "curriculum")
    executor_backend = _build_backend(configs["executor"].get("logging", {}), "executor")

    loop_defaults = _loop_params(configs.get("loop", {}))
    curriculum_batch = args.curriculum_batch or loop_defaults["curriculum_batch"]
    executor_batch = args.executor_batch or loop_defaults["executor_batch"]
    iterations = args.iterations or loop_defaults["iterations"]
    executor_samples = args.executor_samples or loop_defaults["executor_samples"]

    grpo = GRPOTrainer(
        coeffs=_grpo_coeffs(configs["curriculum"].get("rewards", {})),
        backend=curriculum_backend,
        trl_runner=maybe_build_grpo_trl(),
        logger=telemetry,
    )
    adpo = ADPOTrainer(
        lower_clip=0.1,
        base_upper_clip=0.2,
        scale=0.5,
        backend=executor_backend,
        trl_runner=maybe_build_ppo_trl(),
        logger=telemetry,
    )

    loop = CoEvolutionLoop(
        LoopConfig(
            curriculum_batch=curriculum_batch,
            executor_batch=executor_batch,
            iterations=iterations,
            executor_samples=executor_samples,
        ),
        curriculum_agent,
        executor_agent,
        frontier_filter,
        reward_model,
        grpo,
        adpo,
        judge,
    )
    history = loop.run()
    last_frontier = history[-1]["frontier_size"] if history else 0
    telemetry.log({"loop/iterations": len(history), "loop/frontier_last": last_frontier})
    telemetry.close()
    print("Demo iteration complete:", history)


if __name__ == "__main__":
    main()
