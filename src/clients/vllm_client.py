"""HTTP client for vLLM-compatible endpoints."""
from __future__ import annotations

from typing import Any, Dict

import json
import os
import time
from urllib import request


class VLLMHTTPClient:
    """Minimal dependency HTTP client that hits the vLLM REST API."""

    def __init__(self, endpoint: str, api_key_env: str | None = None, timeout: float = 30.0) -> None:
        self._endpoint = endpoint
        self._timeout = timeout
        self._api_key = os.getenv(api_key_env) if api_key_env else None

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self._endpoint, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self._api_key:
            req.add_header("Authorization", f"Bearer {self._api_key}")
        start = time.time()
        with request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
        latency = time.time() - start
        result = json.loads(body)
        result.setdefault("latency", latency)
        return result
