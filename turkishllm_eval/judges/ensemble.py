"""
Multi-Judge Ensemble for TurkishLLM-Eval.

Combines GPT-4o and Claude judgments with weighted averaging,
inter-judge agreement metrics (Cohen's κ), and disagreement resolution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from turkishllm_eval.judges.base import BaseJudge, JudgmentResult
from turkishllm_eval.utils.logger import logger


class JudgeEnsemble:
    """
    Multi-judge ensemble that aggregates evaluations from multiple LLM judges.

    Strategies:
    - weighted_average: Weighted mean of judge scores
    - majority_vote: Binary agree/disagree majority
    - max_agreement: Use score from highest-agreement pair
    """

    def __init__(
        self,
        judges: List[BaseJudge],
        weights: Optional[List[float]] = None,
        strategy: str = "weighted_average",
        agreement_threshold: float = 0.3,
    ):
        self.judges = judges
        self.weights = weights or [1.0 / len(judges)] * len(judges)
        self.strategy = strategy
        self.agreement_threshold = agreement_threshold
        self._judgments_log: List[Dict[str, Any]] = []

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def evaluate(
        self,
        question: str,
        model_response: str,
        reference_answer: str,
        rubric: str = "",
        benchmark_type: str = "general",
    ) -> JudgmentResult:
        """
        Run ensemble evaluation across all judges.

        Returns aggregated JudgmentResult with inter-judge agreement.
        """
        results: List[JudgmentResult] = []

        for judge in self.judges:
            if not judge.is_available():
                logger.warning(f"Judge {judge.name} not available, skipping")
                continue

            try:
                result = judge.judge(
                    question=question,
                    model_response=model_response,
                    reference_answer=reference_answer,
                    rubric=rubric,
                    benchmark_type=benchmark_type,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Judge {judge.name} failed: {e}")

        if not results:
            return JudgmentResult(
                judge_name="ensemble_fallback",
                scores={"overall": 0.5},
                reasoning="No judges available",
                confidence=0.0,
            )

        # Aggregate
        if self.strategy == "weighted_average":
            aggregated = self._weighted_average(results)
        elif self.strategy == "majority_vote":
            aggregated = self._majority_vote(results)
        else:
            aggregated = self._weighted_average(results)

        # Compute agreement
        agreement = self._compute_agreement(results)
        aggregated.metadata["inter_judge_agreement"] = agreement
        aggregated.metadata["num_judges"] = len(results)
        aggregated.metadata["individual_scores"] = {
            r.judge_name: r.primary_score for r in results
        }

        # Log
        self._judgments_log.append({
            "question": question[:100],
            "scores": {r.judge_name: r.primary_score for r in results},
            "agreement": agreement,
            "ensemble_score": aggregated.primary_score,
        })

        return aggregated

    def _weighted_average(self, results: List[JudgmentResult]) -> JudgmentResult:
        """Compute weighted average of judge scores."""
        # Collect all score keys
        all_keys = set()
        for r in results:
            all_keys.update(r.scores.keys())

        # Weighted average per key
        aggregated_scores: Dict[str, float] = {}
        for key in all_keys:
            weighted_sum = 0.0
            weight_sum = 0.0
            for i, r in enumerate(results):
                if key in r.scores:
                    w = self.weights[i] if i < len(self.weights) else 1.0 / len(results)
                    weighted_sum += r.scores[key] * w * r.confidence
                    weight_sum += w * r.confidence
            aggregated_scores[key] = weighted_sum / max(weight_sum, 1e-8)

        # Combine reasoning
        reasonings = [f"[{r.judge_name}] {r.reasoning}" for r in results if r.reasoning]

        return JudgmentResult(
            judge_name="ensemble",
            scores=aggregated_scores,
            reasoning=" | ".join(reasonings),
            confidence=sum(r.confidence for r in results) / len(results),
        )

    def _majority_vote(self, results: List[JudgmentResult]) -> JudgmentResult:
        """Binary majority vote (correct/incorrect at threshold 0.5)."""
        votes = [r.primary_score >= 0.5 for r in results]
        majority = sum(votes) > len(votes) / 2

        overall = sum(r.primary_score for r in results) / len(results) if majority else (
            sum(r.primary_score for r in results) / len(results)
        )

        return JudgmentResult(
            judge_name="ensemble_majority",
            scores={"overall": overall},
            reasoning=f"Majority vote: {sum(votes)}/{len(votes)} positive",
            confidence=abs(sum(votes)/len(votes) - 0.5) * 2,
        )

    def _compute_agreement(self, results: List[JudgmentResult]) -> float:
        """
        Compute inter-judge agreement using score correlation.

        Returns value between 0 (no agreement) and 1 (perfect agreement).
        """
        if len(results) < 2:
            return 1.0

        scores = [r.primary_score for r in results]
        mean = sum(scores) / len(scores)
        deviations = [abs(s - mean) for s in scores]
        max_deviation = max(deviations) if deviations else 0

        # Agreement = 1 - normalized max deviation
        agreement = 1.0 - min(max_deviation / 0.5, 1.0)
        return round(agreement, 4)

    def compute_cohens_kappa(
        self, all_judgments: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """
        Compute Cohen's Kappa for inter-judge agreement over multiple items.

        Uses the logged judgments if no explicit data provided.
        """
        judgments = all_judgments or self._judgments_log
        if len(judgments) < 2 or len(self.judges) < 2:
            return 0.0

        # Binary classification at 0.5 threshold
        judge_names = list(judgments[0].get("scores", {}).keys())
        if len(judge_names) < 2:
            return 0.0

        j1_name, j2_name = judge_names[0], judge_names[1]
        j1_decisions = []
        j2_decisions = []

        for j in judgments:
            scores = j.get("scores", {})
            if j1_name in scores and j2_name in scores:
                j1_decisions.append(1 if scores[j1_name] >= 0.5 else 0)
                j2_decisions.append(1 if scores[j2_name] >= 0.5 else 0)

        if len(j1_decisions) < 2:
            return 0.0

        # Cohen's Kappa calculation
        n = len(j1_decisions)
        agreement = sum(1 for a, b in zip(j1_decisions, j2_decisions) if a == b)
        po = agreement / n  # Observed agreement

        j1_pos = sum(j1_decisions) / n
        j2_pos = sum(j2_decisions) / n
        pe = j1_pos * j2_pos + (1 - j1_pos) * (1 - j2_pos)  # Expected agreement

        if pe == 1.0:
            return 1.0

        kappa = (po - pe) / (1 - pe)
        return round(kappa, 4)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            "strategy": self.strategy,
            "num_judges": len(self.judges),
            "judges": [j.stats for j in self.judges],
            "total_evaluations": len(self._judgments_log),
            "avg_agreement": (
                sum(j.get("agreement", 0) for j in self._judgments_log) / max(len(self._judgments_log), 1)
            ),
            "cohens_kappa": self.compute_cohens_kappa(),
        }
