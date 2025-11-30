"""Thin wrappers around underlying LLM endpoints for curriculum/executor roles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class AgentConfig:
    name: str
    endpoint: str
    max_tokens: int
    temperature: float
    system_prompt: str
    api_key_env: str | None = None


class BaseAgent:
    """Handles prompt assembly and call dispatch to the serving endpoint."""

    def __init__(self, config: AgentConfig, client: Any) -> None:
        self.config = config
        self._client = client

    def build_messages(
        self,
        conversation: List[Dict[str, str]],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.config.name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "system", "content": self.config.system_prompt}] + conversation,
        }
        if extra:
            payload.update(extra)
        return payload

    def invoke_conversation(
        self,
        conversation: List[Dict[str, str]],
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = self.build_messages(conversation, extra)
        response = self._client.invoke(payload)
        return response["choices"][0]["message"]["content"]

    def invoke(self, user_content: str, extra: Optional[Dict[str, Any]] = None) -> str:
        conversation = [{"role": "user", "content": user_content}]
        return self.invoke_conversation(conversation, extra)
