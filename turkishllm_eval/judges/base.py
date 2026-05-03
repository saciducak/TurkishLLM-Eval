"""
Abstract base class for LLM judges in TurkishLLM-Eval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class JudgmentResult:
    """Result from an LLM judge evaluation."""

    judge_name: str
    scores: Dict[str, float]
    reasoning: str
    raw_response: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_score(self) -> float:
        """Get the primary evaluation score."""
        if "overall" in self.scores:
            return self.scores["overall"]
        return sum(self.scores.values()) / max(len(self.scores), 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_name": self.judge_name,
            "scores": self.scores,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BaseJudge(ABC):
    """
    Abstract base class for LLM-as-a-judge implementations.

    Each judge evaluates model responses against reference answers
    using predefined rubrics.
    """

    def __init__(self, name: str, model_id: str, temperature: float = 0.0):
        self.name = name
        self.model_id = model_id
        self.temperature = temperature
        self._call_count = 0
        self._total_tokens = 0

    @abstractmethod
    def judge(
        self,
        question: str,
        model_response: str,
        reference_answer: str,
        rubric: str,
        benchmark_type: str = "general",
    ) -> JudgmentResult:
        """
        Evaluate a model's response.

        Args:
            question: The original question
            model_response: The model's generated answer
            reference_answer: The correct/reference answer
            rubric: Scoring rubric text
            benchmark_type: Type of benchmark for context

        Returns:
            JudgmentResult with scores and reasoning
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this judge's API is available."""
        ...

    @property
    def stats(self) -> Dict[str, Any]:
        """Get judge usage statistics."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }
