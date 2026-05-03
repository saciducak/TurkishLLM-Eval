"""
vLLM model adapter for high-throughput inference.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from turkishllm_eval.models.base import BaseModel, ModelResponse
from turkishllm_eval.utils.logger import logger


class VLLMModel(BaseModel):
    """Adapter for vLLM high-throughput inference engine."""

    def __init__(self, model_id: str, display_name: str = "", tensor_parallel_size: int = 1):
        super().__init__(model_id, display_name)
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM
            self._llm = LLM(model=self.model_id, tensor_parallel_size=self.tensor_parallel_size)
            logger.info(f"vLLM loaded: {self.model_id}")
        except ImportError:
            raise ImportError("vllm package required: pip install vllm")

    def generate(self, prompt: str, system_prompt: str = "", max_new_tokens: int = 512,
                 temperature: float = 0.1, top_p: float = 0.9, **kwargs) -> ModelResponse:
        self._load()
        from vllm import SamplingParams

        start = time.time()
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        params = SamplingParams(temperature=max(temperature, 0.01), top_p=top_p, max_tokens=max_new_tokens)
        outputs = self._llm.generate([full_prompt], params)
        text = outputs[0].outputs[0].text if outputs else ""
        tokens = len(outputs[0].outputs[0].token_ids) if outputs else 0
        self._total_calls += 1
        self._total_tokens += tokens

        return ModelResponse(
            text=text.strip(), model_id=self.model_id,
            tokens_used=tokens, latency_ms=(time.time() - start) * 1000,
        )

    def batch_generate(self, prompts: List[str], system_prompt: str = "",
                       max_new_tokens: int = 512, **kwargs) -> List[ModelResponse]:
        self._load()
        from vllm import SamplingParams

        start = time.time()
        full_prompts = [f"{system_prompt}\n\n{p}" if system_prompt else p for p in prompts]
        params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=max_new_tokens)
        outputs = self._llm.generate(full_prompts, params)

        results = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            tokens = len(out.outputs[0].token_ids) if out.outputs else 0
            results.append(ModelResponse(
                text=text.strip(), model_id=self.model_id,
                tokens_used=tokens, latency_ms=(time.time() - start) * 1000 / len(prompts),
            ))
        self._total_calls += len(prompts)
        return results

    def is_available(self) -> bool:
        try:
            import vllm
            return True
        except ImportError:
            return False
