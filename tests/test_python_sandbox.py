from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tools.python_sandbox import (
    PythonSandbox,
    SandboxConfig,
    StopGoConfig,
    StopGoController,
)


class PythonSandboxTests(unittest.TestCase):
    def test_child_environment_excludes_credentials_and_artifact_stays_local(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir)
            config = SandboxConfig(
                engine="sandbox-cli",
                worker_pool_size=1,
                python_runtime="python3",
                execution_timeout_s=2,
                memory_limit_mb=64,
                artifact_dir=artifact_dir,
            )
            sandbox = PythonSandbox(config)
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"ok\n", stderr=b""
            )

            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "must-not-pass", "PATH": "/usr/bin"},
                clear=True,
            ), patch("src.tools.python_sandbox.subprocess.run", return_value=completed) as run:
                result = sandbox.execute("print('ok')", "../../outside")

            self.assertEqual(result["status"], "ok")
            self.assertEqual(run.call_args.args[0][0], "sandbox-cli")
            self.assertEqual(run.call_args.kwargs["env"], {"PATH": "/usr/bin"})
            artifacts = list(artifact_dir.glob("*.json"))
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].parent, artifact_dir)
            self.assertNotIn("..", artifacts[0].name)
            self.assertEqual(json.loads(artifacts[0].read_text())["stdout"], "ok\n")

    def test_controller_sends_only_python_body_to_sandbox(self) -> None:
        class RecordingSandbox:
            def __init__(self) -> None:
                self.code = ""

            def execute(self, code: str, task_id: str):
                self.code = code
                return {"stdout": "2\n", "stderr": "", "status": "ok"}

        sandbox = RecordingSandbox()
        controller = StopGoController(
            sandbox,  # type: ignore[arg-type]
            StopGoConfig(
                trigger_regex=r"```python\s*\n([\s\S]*?)```",
                max_code_blocks=1,
                capture_stdout=True,
                capture_stderr=True,
            ),
        )

        result = controller.run("```python\nprint(1 + 1)\n```", "example")

        self.assertEqual(sandbox.code, "print(1 + 1)\n")
        self.assertIn("[tool stdout]\n2", result["patched_response"])

    def test_controller_annotates_identical_code_blocks_in_order(self) -> None:
        class RecordingSandbox:
            def __init__(self) -> None:
                self.calls = 0

            def execute(self, code: str, task_id: str):
                self.calls += 1
                return {
                    "stdout": f"result-{self.calls}\n",
                    "stderr": "",
                    "status": "ok",
                }

        sandbox = RecordingSandbox()
        controller = StopGoController(
            sandbox,  # type: ignore[arg-type]
            StopGoConfig(
                trigger_regex=r"```python\s*\n([\s\S]*?)```",
                max_code_blocks=2,
                capture_stdout=True,
                capture_stderr=True,
            ),
        )
        block = "```python\nprint(1)\n```"

        result = controller.run(f"{block}\nthen\n{block}", "duplicate")

        patched = result["patched_response"]
        self.assertIsInstance(patched, str)
        assert isinstance(patched, str)
        self.assertEqual(sandbox.calls, 2)
        self.assertLess(patched.index("result-1"), patched.index("then"))
        self.assertGreater(patched.index("result-2"), patched.index("then"))


if __name__ == "__main__":
    unittest.main()
