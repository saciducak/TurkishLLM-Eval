"""
Ollama local model adapter.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

from turkishllm_eval.models.base import BaseModel, ModelResponse
from turkishllm_eval.utils.logger import logger


class OllamaModel(BaseModel):
    """Adapter for locally running Ollama models."""

    def __init__(
        self,
        model_id: str = "llama3",
        display_name: str = "",
        host: str = "http://localhost:11434",
    ):
        super().__init__(model_id, display_name)
        self.host = host
        self.api_url = f"{host}/api/generate"
        self.chat_url = f"{host}/api/chat"

    def generate(
        self, prompt: str, system_prompt: str = "", max_new_tokens: int = 512,
        temperature: float = 0.1, top_p: float = 0.9, **kwargs,
    ) -> ModelResponse:
        start = time.time()

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens,
            },
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "")
            self._total_calls += 1

            return ModelResponse(
                text=text.strip(), model_id=self.model_id,
                tokens_used=data.get("eval_count", 0),
                latency_ms=(time.time() - start) * 1000,
                finish_reason="stop",
            )
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ModelResponse(text="", model_id=self.model_id, metadata={"error": str(e)})

    def batch_generate(self, prompts: List[str], system_prompt: str = "",
                       max_new_tokens: int = 512, **kwargs) -> List[ModelResponse]:
        return [self.generate(p, system_prompt, max_new_tokens, **kwargs) for p in prompts]

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
