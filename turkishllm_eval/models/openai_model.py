"""
OpenAI API model adapter.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from turkishllm_eval.models.base import BaseModel, ModelResponse
from turkishllm_eval.utils.logger import logger
from turkishllm_eval.utils.rate_limiter import RateLimiter


class OpenAIModel(BaseModel):
    """OpenAI API model adapter (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model_id: str = "gpt-4o", display_name: str = "", api_key: Optional[str] = None):
        super().__init__(model_id, display_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.rate_limiter = RateLimiter(requests_per_minute=50)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, system_prompt: str = "", max_new_tokens: int = 512,
                 temperature: float = 0.1, top_p: float = 0.9, **kwargs) -> ModelResponse:
        self.rate_limiter.wait_if_needed(estimated_tokens=max_new_tokens)
        start = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self.client.chat.completions.create(
                model=self.model_id, messages=messages,
                temperature=temperature, max_tokens=max_new_tokens, top_p=top_p,
            )
            text = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0
            self._total_calls += 1
            self._total_tokens += tokens
            return ModelResponse(
                text=text.strip(), model_id=self.model_id,
                tokens_used=tokens, latency_ms=(time.time() - start) * 1000,
                finish_reason=resp.choices[0].finish_reason or "stop",
            )
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return ModelResponse(text="", model_id=self.model_id, metadata={"error": str(e)})

    def batch_generate(self, prompts: List[str], system_prompt: str = "",
                       max_new_tokens: int = 512, **kwargs) -> List[ModelResponse]:
        return [self.generate(p, system_prompt, max_new_tokens, **kwargs) for p in prompts]

    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0
