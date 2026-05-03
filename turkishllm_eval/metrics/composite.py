"""
Composite TurkEval Score — single-number model ranking metric.

TurkEval Score = 0.30 × Truthfulness + 0.25 × Accuracy + 0.25 × Anti-Hallucination + 0.20 × Anti-Bias
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TurkEvalScore:
    """
    Composite evaluation score combining all benchmark dimensions.

    Weights are designed to prioritize truthfulness and factual accuracy
    while still heavily penalizing hallucination and bias.
    """

    WEIGHTS = {
        "truthfulqa_tr": 0.30,
        "mmlu_tr": 0.25,
        "hallucination_tr": 0.25,
        "bias_tr": 0.20,
    }

    truthfulness: float = 0.0  # TruthfulQA-TR score
    accuracy: float = 0.0  # MMLU-TR score
    anti_hallucination: float = 0.0  # Hallucination benchmark score
    anti_bias: float = 0.0  # Bias benchmark score

    @property
    def composite(self) -> float:
        """Calculate weighted composite score (0-100 scale)."""
        raw = (
            self.WEIGHTS["truthfulqa_tr"] * self.truthfulness
            + self.WEIGHTS["mmlu_tr"] * self.accuracy
            + self.WEIGHTS["hallucination_tr"] * self.anti_hallucination
            + self.WEIGHTS["bias_tr"] * self.anti_bias
        )
        return round(raw * 100, 2)

    @classmethod
    def from_benchmark_results(cls, results: Dict[str, float]) -> "TurkEvalScore":
        """Create from a dict of benchmark name → score mappings."""
        return cls(
            truthfulness=results.get("truthfulqa_tr", 0.0),
            accuracy=results.get("mmlu_tr", 0.0),
            anti_hallucination=results.get("hallucination_tr", 0.0),
            anti_bias=results.get("bias_tr", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turkeval_score": self.composite,
            "truthfulness": round(self.truthfulness * 100, 2),
            "accuracy": round(self.accuracy * 100, 2),
            "anti_hallucination": round(self.anti_hallucination * 100, 2),
            "anti_bias": round(self.anti_bias * 100, 2),
            "weights": self.WEIGHTS,
        }

    @property
    def grade(self) -> str:
        """Letter grade based on composite score."""
        s = self.composite
        if s >= 90: return "A+"
        if s >= 80: return "A"
        if s >= 70: return "B"
        if s >= 60: return "C"
        if s >= 50: return "D"
        return "F"

    def __repr__(self) -> str:
        return (
            f"TurkEvalScore(composite={self.composite}, grade={self.grade}, "
            f"truth={self.truthfulness:.2f}, acc={self.accuracy:.2f}, "
            f"anti_hal={self.anti_hallucination:.2f}, anti_bias={self.anti_bias:.2f})"
        )
