"""
HuggingFace Transformers model adapter for Turkish LLMs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from turkishllm_eval.models.base import BaseModel, ModelResponse
from turkishllm_eval.utils.logger import logger


class HuggingFaceModel(BaseModel):
    """Adapter for HuggingFace transformers models."""

    def __init__(
        self,
        model_id: str,
        display_name: str = "",
        device: str = "auto",
        dtype: str = "float16",
        use_flash_attention: bool = True,
        trust_remote_code: bool = True,
    ):
        super().__init__(model_id, display_name)
        self.device = device
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load model and tokenizer."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading model: {self.model_id}")

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(self.dtype, torch.float16)

            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": self.device,
                "trust_remote_code": self.trust_remote_code,
            }

            if self.use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    pass

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **model_kwargs
            )

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            logger.info(f"Model loaded: {self.model_id}")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_id}: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs,
    ) -> ModelResponse:
        """Generate using transformers pipeline."""
        self._load_model()
        import torch

        start_time = time.time()

        # Format with chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                input_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                input_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        else:
            input_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        inputs = self._tokenizer(input_text, return_tensors="pt")
        if hasattr(self._model, "device"):
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # Decode only new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

        latency = (time.time() - start_time) * 1000
        self._total_calls += 1
        self._total_tokens += len(generated_tokens)

        return ModelResponse(
            text=text.strip(),
            model_id=self.model_id,
            tokens_used=len(generated_tokens),
            latency_ms=latency,
            finish_reason="stop",
        )

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str = "",
        max_new_tokens: int = 512,
        **kwargs,
    ) -> List[ModelResponse]:
        """Batch generation (sequential for simplicity)."""
        return [
            self.generate(p, system_prompt, max_new_tokens, **kwargs)
            for p in prompts
        ]

    def is_available(self) -> bool:
        """Check if transformers is importable."""
        try:
            import transformers
            return True
        except ImportError:
            return False
