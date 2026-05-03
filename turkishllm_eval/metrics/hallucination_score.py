"""Hallucination severity metric."""

from __future__ import annotations
from typing import List, Dict
from turkishllm_eval.benchmarks.base import EvalResult


class HallucinationMetric:
    """Metrics for hallucination detection benchmark."""

    @staticmethod
    def hallucination_rate(results: List[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if not r.is_correct) / len(results)

    @staticmethod
    def avg_severity(results: List[EvalResult]) -> float:
        if not results:
            return 0.0
        return 1.0 - (sum(r.score for r in results) / len(results))

    @staticmethod
    def refusal_accuracy(results: List[EvalResult]) -> float:
        """How well the model refuses to answer unanswerable questions."""
        unanswerable = [r for r in results if r.metadata.get("is_unanswerable", False)]
        if not unanswerable:
            return 1.0
        return sum(
            1 for r in unanswerable if r.metadata.get("shows_uncertainty", False)
        ) / len(unanswerable)

    @staticmethod
    def severity_distribution(results: List[EvalResult]) -> Dict[str, int]:
        dist = {"none": 0, "minor": 0, "major": 0, "critical": 0}
        for r in results:
            sev = r.metadata.get("severity", "none")
            if sev in dist:
                dist[sev] += 1
        return dist
