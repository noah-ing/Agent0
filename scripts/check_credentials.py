#!/usr/bin/env python3
"""Quick credential self-test for verifier + vLLM endpoints."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.settings import load_env

load_env(ROOT)


def _print(status: str, message: str) -> None:
    print(f"[{status.upper()}] {message}")


def _responses_payload(model: str) -> dict:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Answer yes if 2+2=4, otherwise no.",
                    }
                ],
            }
        ],
    }


def _chat_payload(model: str) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a credential checker."},
            {"role": "user", "content": "Does 2+2 equal 4? Reply yes/no."},
        ],
        "max_tokens": 8,
    }


def check_verifier() -> bool:
    endpoint = os.getenv("AGENT0_VERIFIER_ENDPOINT")
    model = os.getenv("AGENT0_VERIFIER_MODEL", "gpt-4o-mini-verifier")
    key = os.getenv("OPENAI_API_KEY")
    if not (endpoint and key):
        _print("warn", "Verifier env vars missing (AGENT0_VERIFIER_ENDPOINT or OPENAI_API_KEY).")
        return False
    if endpoint.rstrip("/").endswith("/responses"):
        payload = _responses_payload(model)
    else:
        payload = _chat_payload(model)
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
        data = json.loads(body)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore") if hasattr(err, "read") else ""
        snippet = detail[:200]
        if err.code in (400, 404) and "does not exist" in detail:
            _print(
                "warn",
                f"Verifier reachable but model '{model}' not available. Update AGENT0_VERIFIER_MODEL. Response: {snippet}",
            )
            return True
        _print("fail", f"Verifier HTTP error: {err.code} {err.reason} {snippet}")
        return False
    except Exception as exc:  # noqa: BLE001
        _print("fail", f"Verifier request failed: {exc}")
        return False
    output = data.get("output")
    if isinstance(output, list) and output:
        verdict = str(output[0])
    elif isinstance(output, str):
        verdict = output
    elif "choices" in data:
        verdict = data["choices"][0]["message"]["content"]
    else:
        verdict = "(unexpected response format)"
    verdict = verdict.replace("\n", " ")[:80]
    _print("ok", f"Verifier reachable via {endpoint} (sample output: {verdict!r})")
    return True


def check_vllm() -> bool:
    base = os.getenv("AGENT0_VLLM_BASE")
    key = os.getenv("AGENT0_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not base:
        _print("warn", "AGENT0_VLLM_BASE not set; skipping vLLM connectivity check.")
        return False
    url = base.rstrip("/") + "/models"
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
        data = json.loads(body)
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore") if hasattr(err, "read") else ""
        _print("fail", f"vLLM HTTP error: {err.code} {err.reason} {detail[:120]}")
        return False
    except Exception as exc:  # noqa: BLE001
        _print("fail", f"vLLM request failed: {exc}")
        return False
    models = data.get("data") or []
    display = models[0].get("id") if models else "(no models returned)"
    _print("ok", f"vLLM endpoint {base} responded (example model: {display})")
    return True


def main() -> None:
    verifier_ok = check_verifier()
    vllm_ok = check_vllm()
    if not (verifier_ok or vllm_ok):
        sys.exit("No credentials validated. See messages above.")


if __name__ == "__main__":
    main()
