"""Helpers for instantiating TRL trainers when the stack is available."""
from __future__ import annotations

from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer as HFGRPOTrainer, PPOConfig, PPOTrainer

    _TRL_AVAILABLE = True
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    GRPOConfig = None  # type: ignore
    HFGRPOTrainer = None  # type: ignore
    PPOConfig = None  # type: ignore
    PPOTrainer = None  # type: ignore
    _TRL_AVAILABLE = False


def build_trl_grpo(model_name: str, grpo_args: Dict[str, Any]) -> Optional[Any]:
    """Return a TRL GRPO trainer configured for curriculum updates."""

    if not _TRL_AVAILABLE:  # pragma: no cover
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = GRPOConfig(**grpo_args)
    trainer = HFGRPOTrainer(model=model, tokenizer=tokenizer, args=config)
    return trainer


def build_trl_ppo(model_name: str, ppo_args: Dict[str, Any]) -> Optional[Any]:
    """Return a TRL PPO trainer (used as ADPO base) if deps are installed."""

    if not _TRL_AVAILABLE:  # pragma: no cover
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    config = PPOConfig(**ppo_args)
    trainer = PPOTrainer(model=model, tokenizer=tokenizer, args=config)
    return trainer
