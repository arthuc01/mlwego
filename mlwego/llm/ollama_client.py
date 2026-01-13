"""Ollama client wrapper."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class OllamaResponse:
    message: Dict[str, Any]
    raw: Dict[str, Any]


class OllamaClient:
    def __init__(self, host: Optional[str] = None, timeout: int = 60) -> None:
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
    ) -> OllamaResponse:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        req = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return OllamaResponse(message=data.get("message", {}), raw=data)

    @staticmethod
    def format_tool_calls(response: OllamaResponse) -> Iterable[Dict[str, Any]]:
        return response.message.get("tool_calls", [])
