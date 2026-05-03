"""Tests for benchmark implementations."""

import pytest
from turkishllm_eval.data.loader import BenchmarkItem, DatasetLoader
from turkishllm_eval.benchmarks.truthfulqa_tr import TruthfulQATR
from turkishllm_eval.benchmarks.mmlu_tr import MMLUTurkish
from turkishllm_eval.benchmarks.hallucination import HallucinationBenchmark
from turkishllm_eval.benchmarks.bias_detection import BiasDetectionBenchmark


class TestTruthfulQATR:
    def test_load_data(self):
        bench = TruthfulQATR()
        items = bench.load_data()
        assert len(items) > 0

    def test_format_prompt_mc(self):
        bench = TruthfulQATR()
        item = BenchmarkItem(
            id="test_1", question="Test soru?", category="tarih",
            correct_answer="Doğru cevap", choices=["A. Doğru", "B. Yanlış", "C. Belki", "D. Hiçbiri"],
            correct_choice_index=0,
        )
        prompt = bench.format_prompt(item, mode="mc")
        assert "Test soru?" in prompt
        assert "A." in prompt

    def test_evaluate_mc_correct(self):
        bench = TruthfulQATR()
        item = BenchmarkItem(
            id="test_2", question="Test?", category="bilim",
            correct_answer="A", choices=["A. Doğru", "B. Yanlış", "C. C", "D. D"],
            correct_choice_index=0,
        )
        result = bench.evaluate_response(item, "A")
        assert result.is_correct
        assert result.score == 1.0

    def test_evaluate_mc_incorrect(self):
        bench = TruthfulQATR()
        item = BenchmarkItem(
            id="test_3", question="Test?", category="bilim",
            correct_answer="A", choices=["A. Doğru", "B. Yanlış", "C. C", "D. D"],
            correct_choice_index=0,
        )
        result = bench.evaluate_response(item, "B")
        assert not result.is_correct


class TestMMLUTurkish:
    def test_load(self):
        bench = MMLUTurkish()
        items = bench.load_data()
        assert len(items) > 0

    def test_evaluate(self):
        bench = MMLUTurkish()
        item = BenchmarkItem(
            id="m1", question="Test?", category="tarih",
            correct_answer="A. Doğru", choices=["A. Doğru", "B. Yanlış", "C. C", "D. D"],
            correct_choice_index=0,
        )
        result = bench.evaluate_response(item, "A")
        assert result.is_correct


class TestHallucination:
    def test_load(self):
        bench = HallucinationBenchmark()
        items = bench.load_data()
        assert len(items) > 0

    def test_uncertainty_detection(self):
        bench = HallucinationBenchmark()
        assert bench._check_uncertainty("Bu konuda bilmiyorum.")
        assert not bench._check_uncertainty("Kesinlikle doğrudur.")


class TestBiasDetection:
    def test_load(self):
        bench = BiasDetectionBenchmark()
        items = bench.load_data()
        assert len(items) > 0

    def test_stereotype_check(self):
        bench = BiasDetectionBenchmark()
        assert bench._check_stereotypes("kadınlar yapamaz bu işi", "cinsiyet")
        assert not bench._check_stereotypes("herkes eşittir", "cinsiyet")
