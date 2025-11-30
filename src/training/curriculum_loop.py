"""High-level orchestration for the Agent0 co-evolution loop."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .adpo import AmbiguityStats
from ..types import EvaluatedSample, ExecutorFeedback


@dataclass
class LoopConfig:
    curriculum_batch: int
    executor_batch: int
    iterations: int
    executor_samples: int = 4


class CoEvolutionLoop:
    """Alternates curriculum updates and executor ADPO steps."""

    def __init__(
        self,
        cfg: LoopConfig,
        curriculum_agent: Any,
        executor_agent: Any,
        frontier_filter: Any,
        reward_model: Any,
        grpo_trainer: Any,
        adpo_trainer: Any,
        judge: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.curriculum_agent = curriculum_agent
        self.executor_agent = executor_agent
        self.frontier_filter = frontier_filter
        self.reward_model = reward_model
        self.grpo_trainer = grpo_trainer
        self.adpo_trainer = adpo_trainer
        self.judge = judge

    def _normalize_answer(self, text: str) -> str:
        clean = text.strip()
        boxed = re.findall(r"\\boxed\{([^}]*)\}", clean)
        if boxed:
            return boxed[-1].strip()
        lower = clean.lower()
        if "final answer" in lower:
            idx = lower.rfind("final answer")
            snippet = clean[idx + len("final answer") :]
            snippet = snippet.lstrip(": ").strip()
            return snippet
        if "</final_answer>" in lower:
            snippet = clean.split("</final_answer>")[0]
            return snippet.strip()
        return clean

    def _trace_to_stats(self, prompt: str, trace: Any, feedback: ExecutorFeedback | None) -> AmbiguityStats:
        consistency = feedback.p_hat if feedback else 0.5
        normalized = self._normalize_answer(trace.final_answer)
        is_majority = bool(feedback and normalized == feedback.majority_answer)
        turn_penalty = min(0.3, 0.04 * max(0, len(trace.turns) - 2))
        tool_penalty = min(0.2, 0.02 * len(trace.tool_events))
        consistency = max(0.0, min(1.0, consistency - turn_penalty - tool_penalty))
        advantage = 1.0 if is_majority else -0.25
        advantage -= 0.01 * len(trace.tool_events)
        return AmbiguityStats(
            consistency=consistency,
            advantage=advantage,
            prompt=prompt,
            response=trace.final_answer,
        )

    def _collect_executor_feedback(self, task: str) -> ExecutorFeedback:
        traces: List[Any] = []
        answers: List[str] = []
        tool_counts: List[int] = []
        samples = max(1, self.cfg.executor_samples)
        for _ in range(samples):
            trace = self.executor_agent.solve(task)
            traces.append(trace)
            normalized = self._normalize_answer(trace.final_answer)
            answers.append(normalized)
            tool_counts.append(len(trace.tool_events))
        if answers:
            majority_answer, count = Counter(answers).most_common(1)[0]
            p_hat = count / len(answers)
        else:
            majority_answer = ""
            p_hat = 0.0
        return ExecutorFeedback(
            p_hat=p_hat,
            majority_answer=majority_answer,
            tool_counts=tool_counts,
            traces=traces,
            answers=answers,
        )

    def run_iteration(self) -> Dict[str, Any]:
        curriculum_batch = self.curriculum_agent.generate_batch(self.cfg.curriculum_batch)
        evaluated: List[EvaluatedSample] = []
        reward_payload: List[Dict[str, Any]] = []
        for sample in curriculum_batch:
            feedback = self._collect_executor_feedback(sample.raw_output)
            stats_payload = {
                "p_hat": feedback.p_hat,
                "avg_tool_calls": feedback.avg_tool_calls,
            }
            rewards = self.reward_model.score(sample.raw_output, stats_payload)
            reward_payload.append(
                {
                    "prompt": sample.prompt,
                    "response": sample.raw_output,
                    "reward_breakdown": rewards,
                }
            )
            evaluated.append(EvaluatedSample(sample=sample, feedback=feedback))
        self.grpo_trainer.step(reward_payload)

        frontier = self.frontier_filter.build_frontier(evaluated)
        exec_stats: List[Any] = []
        ambiguity_batch: List[AmbiguityStats] = []
        for record in frontier[: self.cfg.executor_batch]:
            feedback = record.feedback
            if not feedback:
                continue
            for trace in feedback.traces:
                exec_stats.append(trace)
                ambiguity_batch.append(self._trace_to_stats(record.sample.raw_output, trace, feedback))
        if ambiguity_batch:
            self.adpo_trainer.step(ambiguity_batch)
        return {"frontier_size": len(frontier), "executor_traces": exec_stats}

    def run(self) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for _ in range(self.cfg.iterations):
            history.append(self.run_iteration())
        return history
