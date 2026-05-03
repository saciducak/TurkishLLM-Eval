"""Tests for metrics."""

import pytest
from turkishllm_eval.metrics.composite import TurkEvalScore
from turkishllm_eval.metrics.agreement import AgreementMetric


class TestTurkEvalScore:
    def test_composite(self):
        score = TurkEvalScore(truthfulness=0.8, accuracy=0.7, anti_hallucination=0.75, anti_bias=0.6)
        assert 0 < score.composite <= 100

    def test_grade(self):
        assert TurkEvalScore(0.95, 0.95, 0.95, 0.95).grade == "A+"
        assert TurkEvalScore(0.3, 0.3, 0.3, 0.3).grade == "F"

    def test_from_dict(self):
        score = TurkEvalScore.from_benchmark_results({
            "truthfulqa_tr": 0.8, "mmlu_tr": 0.7,
            "hallucination_tr": 0.75, "bias_tr": 0.65,
        })
        assert score.composite > 0


class TestAgreement:
    def test_perfect_agreement(self):
        kappa = AgreementMetric.cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0])
        assert kappa == 1.0

    def test_no_agreement(self):
        kappa = AgreementMetric.cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1])
        assert kappa < 0

    def test_interpret(self):
        assert "perfect" in AgreementMetric.interpret_kappa(0.9).lower()
        assert "slight" in AgreementMetric.interpret_kappa(0.1).lower()
