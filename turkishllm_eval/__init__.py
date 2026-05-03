"""
TurkishLLM-Eval — Comprehensive Hallucination, Factual Accuracy & Bias Benchmark Suite
for Turkish Large Language Models.

A first-of-its-kind open-source evaluation framework designed specifically for Turkish LLMs,
featuring LLM-as-a-judge ensemble pipelines, culturally-adapted benchmarks, and a live
Hugging Face Spaces leaderboard.

Usage:
    from turkishllm_eval import EvalPipeline, TruthfulQATR, MMLUTurkish

    pipeline = EvalPipeline(config="configs/default.yaml")
    results = pipeline.run(model="trendyol/Trendyol-LLM-7b-chat")
"""

__version__ = "0.1.0"
__author__ = "Muhammed Sacid Ucak"
__license__ = "Apache-2.0"

from turkishllm_eval.config import EvalConfig, load_config
from turkishllm_eval.pipeline import EvalPipeline

__all__ = [
    "EvalConfig",
    "EvalPipeline",
    "load_config",
    "__version__",
]
