"""
Main evaluation pipeline for TurkishLLM-Eval.

Orchestrates: Model Loading → Benchmark Execution → Judge Evaluation → Scoring → Export.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from turkishllm_eval.benchmarks import (
    BaseBenchmark, BiasDetectionBenchmark, HallucinationBenchmark, MMLUTurkish, TruthfulQATR,
)
from turkishllm_eval.config import EvalConfig, TURKISH_MODEL_REGISTRY
from turkishllm_eval.judges import ClaudeJudge, GPT4Judge, JudgeEnsemble
from turkishllm_eval.metrics.composite import TurkEvalScore
from turkishllm_eval.models.base import BaseModel
from turkishllm_eval.utils.export import ResultExporter
from turkishllm_eval.utils.logger import logger, log_benchmark_start, log_benchmark_result


BENCHMARK_REGISTRY = {
    "truthfulqa_tr": TruthfulQATR,
    "mmlu_tr": MMLUTurkish,
    "hallucination_tr": HallucinationBenchmark,
    "bias_tr": BiasDetectionBenchmark,
}


class EvalPipeline:
    """
    Main evaluation pipeline.

    Usage:
        config = load_config("configs/default.yaml")
        pipeline = EvalPipeline(config)
        results = pipeline.run()
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.model: Optional[BaseModel] = None
        self.judge_ensemble: Optional[JudgeEnsemble] = None
        self.exporter = ResultExporter(config.output_dir)

    def _init_model(self) -> BaseModel:
        """Initialize the target model based on config."""
        model_info = self.config.resolve_model()
        model_type = model_info.get("type", self.config.model_type)

        if model_type == "huggingface":
            from turkishllm_eval.models.huggingface_model import HuggingFaceModel
            return HuggingFaceModel(
                model_id=model_info["model_id"],
                display_name=model_info.get("display_name", ""),
                device=self.config.device,
                dtype=self.config.dtype,
            )
        elif model_type == "openai":
            from turkishllm_eval.models.openai_model import OpenAIModel
            return OpenAIModel(
                model_id=model_info["model_id"],
                display_name=model_info.get("display_name", ""),
                api_key=self.config.openai_api_key,
            )
        elif model_type == "ollama":
            from turkishllm_eval.models.ollama_model import OllamaModel
            return OllamaModel(
                model_id=model_info["model_id"],
                display_name=model_info.get("display_name", ""),
            )
        elif model_type == "vllm":
            from turkishllm_eval.models.vllm_model import VLLMModel
            return VLLMModel(
                model_id=model_info["model_id"],
                display_name=model_info.get("display_name", ""),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _init_judges(self) -> JudgeEnsemble:
        """Initialize judge ensemble."""
        judges = []
        weights = []

        gpt4 = GPT4Judge(
            model_id=self.config.judge.primary_judge,
            api_key=self.config.openai_api_key,
        )
        if gpt4.is_available():
            judges.append(gpt4)
            weights.append(self.config.judge.primary_weight)

        claude = ClaudeJudge(
            model_id=self.config.judge.secondary_judge,
            api_key=self.config.anthropic_api_key,
        )
        if claude.is_available():
            judges.append(claude)
            weights.append(self.config.judge.secondary_weight)

        if not judges:
            logger.warning("No judge APIs available — using heuristic evaluation only")
            return None

        return JudgeEnsemble(
            judges=judges,
            weights=weights,
            strategy=self.config.judge.ensemble_strategy,
        )

    def run(
        self,
        benchmarks: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full evaluation pipeline.

        Args:
            benchmarks: List of benchmark names to run. None = all configured.
            max_samples: Override max samples per benchmark.

        Returns:
            Complete evaluation results dictionary.
        """
        logger.info("━" * 60)
        logger.info("🇹🇷 TurkishLLM-Eval — Starting Evaluation Pipeline")
        logger.info("━" * 60)

        # Initialize
        self.model = self._init_model()
        self.judge_ensemble = self._init_judges()

        model_info = self.config.resolve_model()
        logger.info(f"Model: {model_info.get('display_name', self.config.model_name)}")
        logger.info(f"Judge ensemble: {'Active' if self.judge_ensemble else 'Disabled (heuristic mode)'}")

        # Select benchmarks
        benchmark_names = benchmarks or self.config.benchmark.benchmarks
        max_samples = max_samples or self.config.benchmark.max_samples

        results = {
            "model_name": self.config.model_name,
            "model_id": model_info.get("model_id", ""),
            "display_name": model_info.get("display_name", ""),
            "developer": model_info.get("developer", "Unknown"),
            "parameters": model_info.get("parameters", "Unknown"),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "benchmarks": benchmark_names,
                "max_samples": max_samples,
                "judge_strategy": self.config.judge.ensemble_strategy,
            },
            "benchmarks": {},
        }

        # Run each benchmark
        benchmark_scores = {}
        for bname in benchmark_names:
            if bname not in BENCHMARK_REGISTRY:
                logger.warning(f"Unknown benchmark: {bname}, skipping")
                continue

            benchmark = BENCHMARK_REGISTRY[bname](
                num_few_shot=self.config.benchmark.num_few_shot
            )
            bench_results = self._run_benchmark(benchmark, max_samples)
            results["benchmarks"][bname] = bench_results.to_dict()
            benchmark_scores[bname] = bench_results.score

        # Composite score
        turkeval = TurkEvalScore.from_benchmark_results(benchmark_scores)
        results["turkeval_score"] = turkeval.composite
        results["turkeval_grade"] = turkeval.grade
        results["turkeval_breakdown"] = turkeval.to_dict()

        # Judge stats
        if self.judge_ensemble:
            results["judge_stats"] = self.judge_ensemble.stats

        # Export
        self.exporter.export_json(results)
        self.exporter.export_leaderboard_entry(results)

        logger.info("━" * 60)
        logger.info(f"🏆 TurkEval Score: {turkeval.composite} ({turkeval.grade})")
        logger.info("━" * 60)

        return results

    def _run_benchmark(self, benchmark: BaseBenchmark, max_samples: Optional[int]):
        """Run a single benchmark."""
        from turkishllm_eval.benchmarks.base import BenchmarkResults, EvalResult

        log_benchmark_start(logger, benchmark.name, self.config.model_name)

        items = benchmark.load_data(max_samples=max_samples)
        if not items:
            logger.warning(f"No data loaded for {benchmark.name}")
            return BenchmarkResults(
                benchmark_name=benchmark.name, model_name=self.config.model_name,
                score=0.0, num_samples=0, num_correct=0,
            )

        eval_results: List[EvalResult] = []

        for i, item in enumerate(items):
            prompt = benchmark.format_prompt(item)

            # Get model response
            try:
                response = self.model.generate(
                    prompt=prompt,
                    system_prompt=getattr(benchmark, "SYSTEM_PROMPT", ""),
                    **self.config.benchmark.generation_kwargs,
                )
                model_text = response.text
            except Exception as e:
                logger.error(f"Model generation failed for {item.id}: {e}")
                model_text = ""

            # Judge evaluation
            judge_result = None
            if self.judge_ensemble and model_text:
                try:
                    judgment = self.judge_ensemble.evaluate(
                        question=item.question,
                        model_response=model_text,
                        reference_answer=item.correct_answer,
                        benchmark_type=benchmark.name,
                    )
                    judge_result = judgment.scores
                except Exception as e:
                    logger.debug(f"Judge failed for {item.id}: {e}")

            # Evaluate
            result = benchmark.evaluate_response(item, model_text, judge_result)
            result.metadata["model_name"] = self.config.model_name
            eval_results.append(result)

            if (i + 1) % 10 == 0:
                running_score = sum(r.score for r in eval_results) / len(eval_results)
                logger.info(f"  Progress: {i+1}/{len(items)} | Running score: {running_score:.4f}")

        # Aggregate
        aggregated = benchmark.compute_aggregate(eval_results)
        log_benchmark_result(logger, benchmark.name, aggregated.score, aggregated.num_samples)

        return aggregated
