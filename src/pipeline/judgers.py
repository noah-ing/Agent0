"""Adapters for GPT-4o Mini verifier or local graders."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Optional
from urllib import request


@dataclass
class JudgeResult:
    is_valid: bool
    feedback: str


class GPT4MiniJudge:
    """Calls remote verifier when available, otherwise applies format heuristics."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-4o-mini-verifier",
    ) -> None:
        self._endpoint = endpoint or os.getenv("AGENT0_VERIFIER_ENDPOINT")
        self._api_key = os.getenv(api_key_env)
        self._model = os.getenv("AGENT0_VERIFIER_MODEL", model)

    def _remote_verify(self, sample: str) -> Optional[JudgeResult]:
        if not (self._endpoint and self._api_key):
            return None
        payload = {
            "model": self._model,
            "input": sample,
            "instructions": "Return `yes` if the answer is correct and well formatted, else `no`.",
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self._endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self._api_key}")
        try:
            with request.urlopen(req, timeout=15) as resp:  # noqa: S310
                body = resp.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            return JudgeResult(is_valid=False, feedback=f"verifier error: {exc}")
        result = json.loads(body)
        verdict = str(result.get("output", "no")).strip().lower()
        return JudgeResult(is_valid=verdict.startswith("y"), feedback="remote")

    def verify(self, sample: str) -> JudgeResult:
        remote = self._remote_verify(sample)
        if remote:
            return remote
        if "\boxed" in sample:
            return JudgeResult(is_valid=True, feedback="format ok (local)")
        return JudgeResult(is_valid=False, feedback="missing boxed answer (local)")
