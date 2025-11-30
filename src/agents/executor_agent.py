"""Executor agent that solves tasks via multi-turn tool use."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base_agent import AgentConfig, BaseAgent

if TYPE_CHECKING:
    from ..utils.telemetry import TelemetryLogger


@dataclass
class TurnRecord:
    role: str
    content: str
    tool_events: Optional[List[Dict[str, Any]]] = None


@dataclass
class ExecutionTrace:
    task: str
    turns: List[TurnRecord]
    transcript: str
    tool_events: List[Dict[str, Any]]
    final_answer: str
    rollout_path: Path


class ExecutorAgent(BaseAgent):
    """Wraps stop-go orchestration and captures traces for ADPO training."""

    def __init__(
        self,
        config: AgentConfig,
        client_factory: Any,
        tool_controller: Any,
        max_turns: int = 4,
        rollout_dir: Optional[Path] = None,
        continuation_prompt: Optional[str] = None,
        final_markers: Optional[List[str]] = None,
        logger: Optional["TelemetryLogger"] = None,
    ) -> None:
        client = client_factory(config)
        super().__init__(config, client)
        self._tool = tool_controller
        self._max_turns = max_turns
        self._rollout_dir = rollout_dir or Path("data/rollouts")
        self._rollout_dir.mkdir(parents=True, exist_ok=True)
        self._continuation_template = continuation_prompt or (
            "Continue the reasoning for turn {turn}. "
            "Incorporate any tool output above before proceeding. "
            "When you are certain, respond with 'FINAL ANSWER: <value>'."
        )
        self._final_markers = final_markers or ["FINAL ANSWER:", "\\boxed{", "</final_answer>"]
        self._telemetry = logger

    def _is_terminal(self, text: str) -> bool:
        return any(marker in text for marker in self._final_markers)

    def _persist(self, trace_id: str, payload: Dict[str, Any]) -> Path:
        timestamp = int(time.time() * 1000)
        path = self._rollout_dir / f"{trace_id}_{timestamp}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def solve(self, task: str) -> ExecutionTrace:
        trace_id = f"exec_{abs(hash(task))}"
        conversation: List[Dict[str, str]] = [{"role": "user", "content": task}]
        turns: List[TurnRecord] = [TurnRecord(role="user", content=task)]
        aggregated_events: List[Dict[str, Any]] = []
        final_answer = ""

        for turn_idx in range(self._max_turns):
            response = self.invoke_conversation(conversation)
            tool_payload = self._tool.run(response, task_id=f"{trace_id}_turn{turn_idx}")
            patched = tool_payload.get("patched_response", response)
            turn_events = tool_payload.get("tool_events") or []
            aggregated_events.extend(turn_events)
            turns.append(TurnRecord(role="assistant", content=patched, tool_events=turn_events))
            if self._telemetry:
                self._telemetry.log(
                    {
                        "executor/tool_events": len(turn_events),
                        "executor/turn_index": turn_idx + 1,
                        "executor/turn_tokens": len(patched),
                    }
                )

            if self._is_terminal(patched) or turn_idx == self._max_turns - 1:
                final_answer = patched
                conversation.append({"role": "assistant", "content": patched})
                break

            conversation.append({"role": "assistant", "content": patched})
            follow_up = self._continuation_template.format(turn=turn_idx + 2, events=len(turn_events))
            conversation.append({"role": "user", "content": follow_up})
            turns.append(TurnRecord(role="user", content=follow_up))

        if not final_answer:
            final_answer = turns[-1].content

        transcript = "\n\n".join(
            f"[{idx}] {turn.role}: {turn.content}" for idx, turn in enumerate(turns)
        )
        rollout_payload = {
            "task": task,
            "trace_id": trace_id,
            "turns": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "tool_events": turn.tool_events,
                }
                for turn in turns
            ],
            "final_answer": final_answer,
            "tool_events": aggregated_events,
        }
        rollout_path = self._persist(trace_id, rollout_payload)
        if self._telemetry:
            self._telemetry.log_rollout(
                {
                    "task": task,
                    "trace_id": trace_id,
                    "turns": len(turns),
                    "tool_events": len(aggregated_events),
                }
            )

        return ExecutionTrace(
            task=task,
            turns=turns,
            transcript=transcript,
            tool_events=aggregated_events,
            final_answer=final_answer,
            rollout_path=rollout_path,
        )
