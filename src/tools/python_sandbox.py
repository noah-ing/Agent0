"""SandFuzz-based Python sandbox with stop-go orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import json
import os
import re
import subprocess
import time


_SANDBOX_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SYSTEMROOT")
_UNSAFE_TASK_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _sandbox_environment() -> Dict[str, str]:
    """Return the small, non-secret environment passed to the sandbox CLI."""
    return {name: os.environ[name] for name in _SANDBOX_ENV_ALLOWLIST if name in os.environ}


def _safe_artifact_id(task_id: str) -> str:
    """Keep artifact names inside the configured directory."""
    normalized = _UNSAFE_TASK_ID_CHARS.sub("_", task_id).strip("._")
    return (normalized or "task")[:96]


@dataclass
class SandboxConfig:
    engine: str
    worker_pool_size: int
    python_runtime: str
    execution_timeout_s: int
    memory_limit_mb: int
    artifact_dir: Path
    log_path: Optional[Path] = None


@dataclass
class StopGoConfig:
    trigger_regex: str
    max_code_blocks: int
    capture_stdout: bool
    capture_stderr: bool


class SandboxResult(TypedDict):
    task_id: str
    code: str
    stdout: str
    stderr: str
    status: str
    latency_s: float


class StopGoResult(TypedDict):
    patched_response: str
    tool_events: Optional[List[SandboxResult]]


class PythonSandbox:
    """Delegates code blocks to SandFuzz CLI and captures stdout/stderr."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        if self.config.log_path:
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_payload(self, payload: SandboxResult) -> None:
        if not self.config.log_path:
            return
        with self.config.log_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(payload) + "\n")

    def execute(self, code: str, task_id: str) -> SandboxResult:
        cmd = [
            self.config.engine,
            "run",
            "--lang",
            self.config.python_runtime,
            "--timeout",
            str(self.config.execution_timeout_s),
            "--memory",
            str(self.config.memory_limit_mb),
        ]
        start = time.time()
        try:
            proc = subprocess.run(  # noqa: PLW1510
                cmd,
                input=code.encode("utf-8"),
                capture_output=True,
                timeout=self.config.execution_timeout_s + 2,
                check=False,
                env=_sandbox_environment(),
            )
            status = "ok" if proc.returncode == 0 else f"exit-{proc.returncode}"
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
        except FileNotFoundError:
            status = "missing-binary"
            stdout = ""
            stderr = f"{self.config.engine} binary not found"
        except subprocess.TimeoutExpired:
            status = "timeout"
            stdout = ""
            stderr = "execution exceeded timeout"
        latency = time.time() - start
        payload: SandboxResult = {
            "task_id": task_id,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "status": status,
            "latency_s": latency,
        }
        artifact_path = self.config.artifact_dir / f"{_safe_artifact_id(task_id)}.json"
        artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._log_payload(payload)
        return payload


class StopGoController:
    """Parses model output, runs code blocks, and feeds results back."""

    def __init__(self, sandbox: PythonSandbox, cfg: StopGoConfig) -> None:
        self._sandbox = sandbox
        self._cfg = cfg
        self._regex = re.compile(self._cfg.trigger_regex, re.MULTILINE)

    def run(self, response: str, task_id: str) -> StopGoResult:
        matches = list(self._regex.finditer(response))[: self._cfg.max_code_blocks]
        if not matches:
            return {"patched_response": response, "tool_events": None}
        patched_parts: List[str] = []
        cursor = 0
        events: List[SandboxResult] = []
        for idx, match in enumerate(matches):
            patched_parts.append(response[cursor : match.end()])
            code = match.group(1) if match.groups() else match.group(0)
            exec_id = f"{task_id}_code{idx}"
            result = self._sandbox.execute(code, exec_id)
            events.append(result)
            snippets = []
            if self._cfg.capture_stdout and result.get("stdout"):
                snippets.append(f"[tool stdout]\n{result['stdout']}")
            if self._cfg.capture_stderr and result.get("stderr"):
                snippets.append(f"[tool stderr]\n{result['stderr']}")
            trailer = "\n\n" + "\n".join(snippets) if snippets else ""
            patched_parts.append(trailer)
            cursor = match.end()
        patched_parts.append(response[cursor:])
        patched = "".join(patched_parts)
        return {"patched_response": patched, "tool_events": events}
