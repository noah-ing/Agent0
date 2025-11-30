"""OpenCompass model shim for the Agent0 executor endpoint."""
from __future__ import annotations

import os

from opencompass.models import OpenAISDK

if hasattr(os, "build"):
    os = os.build()

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
)

model_name = os.getenv("AGENT0_EVAL_MODEL", "agent0-executor")
api_key = os.getenv("AGENT0_EVAL_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY"))
base_url = os.getenv("AGENT0_VLLM_BASE", "http://localhost:8002/v1")
if not base_url:
    raise ValueError("AGENT0_VLLM_BASE (OpenAI-compatible endpoint) must be set for OpenCompass runs")

models = [
    dict(
        abbr="agent0-vllm",
        type=OpenAISDK,
        path=model_name,
        key=api_key,
        openai_api_base=base_url,
        temperature=float(os.getenv("AGENT0_EVAL_TEMP", "0.0")),
        meta_template=api_meta_template,
        query_per_second=float(os.getenv("AGENT0_EVAL_QPS", "0.5")),
        max_out_len=int(os.getenv("AGENT0_EVAL_MAX_OUT", "1024")),
        max_seq_len=int(os.getenv("AGENT0_EVAL_MAX_SEQ", "4096")),
        batch_size=int(os.getenv("AGENT0_EVAL_BATCH", "1")),
        retry=int(os.getenv("AGENT0_EVAL_RETRIES", "3")),
    ),
]
