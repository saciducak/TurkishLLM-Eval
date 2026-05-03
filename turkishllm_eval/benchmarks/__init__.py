"""Benchmark subpackage."""

from turkishllm_eval.benchmarks.base import BaseBenchmark, EvalResult
from turkishllm_eval.benchmarks.truthfulqa_tr import TruthfulQATR
from turkishllm_eval.benchmarks.mmlu_tr import MMLUTurkish
from turkishllm_eval.benchmarks.hallucination import HallucinationBenchmark
from turkishllm_eval.benchmarks.bias_detection import BiasDetectionBenchmark

__all__ = [
    "BaseBenchmark",
    "EvalResult",
    "TruthfulQATR",
    "MMLUTurkish",
    "HallucinationBenchmark",
    "BiasDetectionBenchmark",
]
