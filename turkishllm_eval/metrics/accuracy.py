"""Factual accuracy metric."""

from __future__ import annotations
from typing import List
from turkishllm_eval.benchmarks.base import EvalResult


class AccuracyMetric:
    """Standard accuracy metric for MC benchmarks."""

    @staticmethod
    def compute(results: List[EvalResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.is_correct) / len(results)

    @staticmethod
    def compute_per_category(results: List[EvalResult]) -> dict:
        cats = {}
        for r in results:
            if r.category not in cats:
                cats[r.category] = {"correct": 0, "total": 0}
            cats[r.category]["total"] += 1
            if r.is_correct:
                cats[r.category]["correct"] += 1
        return {
            c: d["correct"] / d["total"] if d["total"] > 0 else 0
            for c, d in cats.items()
        }
