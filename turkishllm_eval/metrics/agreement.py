"""Inter-judge agreement metrics."""

from __future__ import annotations
from typing import List, Tuple
import math


class AgreementMetric:
    """Compute inter-judge agreement statistics."""

    @staticmethod
    def cohens_kappa(judge1: List[int], judge2: List[int]) -> float:
        """
        Cohen's Kappa for binary classifications.

        Args:
            judge1: Binary decisions from judge 1 (0 or 1)
            judge2: Binary decisions from judge 2 (0 or 1)

        Returns:
            Cohen's Kappa (-1 to 1, 1 = perfect agreement)
        """
        if len(judge1) != len(judge2) or len(judge1) == 0:
            return 0.0

        n = len(judge1)
        agreement = sum(1 for a, b in zip(judge1, judge2) if a == b)
        po = agreement / n

        p1 = sum(judge1) / n
        p2 = sum(judge2) / n
        pe = p1 * p2 + (1 - p1) * (1 - p2)

        if pe == 1.0:
            return 1.0

        return (po - pe) / (1 - pe)

    @staticmethod
    def score_correlation(scores1: List[float], scores2: List[float]) -> float:
        """Pearson correlation between two sets of scores."""
        if len(scores1) != len(scores2) or len(scores1) < 2:
            return 0.0

        n = len(scores1)
        mean1 = sum(scores1) / n
        mean2 = sum(scores2) / n

        cov = sum((a - mean1) * (b - mean2) for a, b in zip(scores1, scores2)) / n
        std1 = math.sqrt(sum((a - mean1) ** 2 for a in scores1) / n)
        std2 = math.sqrt(sum((b - mean2) ** 2 for b in scores2) / n)

        if std1 == 0 or std2 == 0:
            return 0.0

        return cov / (std1 * std2)

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """Interpret Cohen's Kappa value."""
        if kappa < 0:
            return "Poor (worse than chance)"
        elif kappa < 0.20:
            return "Slight agreement"
        elif kappa < 0.40:
            return "Fair agreement"
        elif kappa < 0.60:
            return "Moderate agreement"
        elif kappa < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"
