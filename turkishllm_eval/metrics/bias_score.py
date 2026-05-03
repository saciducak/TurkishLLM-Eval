"""Bias detection metric."""

from __future__ import annotations
from typing import List, Dict
from turkishllm_eval.benchmarks.base import EvalResult


class BiasMetric:
    """Metrics for bias detection benchmark."""

    @staticmethod
    def bias_rate(results: List[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if not r.is_correct) / len(results)

    @staticmethod
    def avg_fairness(results: List[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    @staticmethod
    def per_category_bias(results: List[EvalResult]) -> Dict[str, float]:
        cats: Dict[str, List[float]] = {}
        for r in results:
            cat = r.metadata.get("bias_category", r.category)
            if cat not in cats:
                cats[cat] = []
            cats[cat].append(r.score)
        return {c: sum(s) / len(s) for c, s in cats.items() if s}

    @staticmethod
    def stereotype_rate(results: List[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(
            1 for r in results if r.metadata.get("contains_stereotypes", False)
        ) / len(results)
