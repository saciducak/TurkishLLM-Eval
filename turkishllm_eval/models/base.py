"""
Abstract base class for model adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelResponse:
    """Response from a model inference call."""

    text: str
    model_id: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, model_id: str, display_name: str = ""):
        self.model_id = model_id
        self.display_name = display_name or model_id
        self._total_calls = 0
        self._total_tokens = 0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        **kwargs,
    ) -> ModelResponse:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: str = "",
        max_new_tokens: int = 512,
        **kwargs,
    ) -> List[ModelResponse]:
        """Generate text for multiple prompts."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available for inference."""
        ...

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
        }
