"""
Abstract base class for all TurkishLLM-Eval benchmarks.

Each benchmark must implement loading data, formatting prompts,
and evaluating model responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from turkishllm_eval.data.loader import BenchmarkItem, DatasetLoader


@dataclass
class EvalResult:
    """Result of evaluating a single benchmark item."""

    item_id: str
    question: str
    model_response: str
    score: float  # 0.0 - 1.0
    is_correct: bool
    category: str
    judge_scores: Dict[str, float] = field(default_factory=dict)
    judge_reasoning: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "model_response": self.model_response,
            "score": self.score,
            "is_correct": self.is_correct,
            "category": self.category,
            "judge_scores": self.judge_scores,
            "judge_reasoning": self.judge_reasoning,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResults:
    """Aggregated results for an entire benchmark run."""

    benchmark_name: str
    model_name: str
    score: float  # Overall score (0.0 - 1.0)
    num_samples: int
    num_correct: int
    category_scores: Dict[str, float] = field(default_factory=dict)
    individual_results: List[EvalResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Overall accuracy as a percentage."""
        return (self.num_correct / self.num_samples * 100) if self.num_samples > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "model_name": self.model_name,
            "score": self.score,
            "accuracy_pct": self.accuracy,
            "num_samples": self.num_samples,
            "num_correct": self.num_correct,
            "category_scores": self.category_scores,
            "metadata": self.metadata,
            "individual_results": [r.to_dict() for r in self.individual_results],
        }


class BaseBenchmark(ABC):
    """
    Abstract base class for evaluation benchmarks.

    All benchmarks (TruthfulQA-TR, MMLU-TR, Hallucination, Bias)
    inherit from this class and implement the core interface.
    """

    def __init__(
        self,
        name: str,
        description: str,
        data_dir: Optional[str] = None,
        num_few_shot: int = 0,
    ):
        self.name = name
        self.description = description
        self.loader = DatasetLoader(data_dir)
        self.num_few_shot = num_few_shot
        self._items: Optional[List[BenchmarkItem]] = None

    @abstractmethod
    def format_prompt(self, item: BenchmarkItem) -> str:
        """
        Format a benchmark item into a prompt for the model.

        Args:
            item: Benchmark item to format.

        Returns:
            Formatted prompt string.
        """
        ...

    @abstractmethod
    def evaluate_response(
        self,
        item: BenchmarkItem,
        model_response: str,
        judge_result: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """
        Evaluate a model's response to a benchmark item.

        Args:
            item: The original benchmark item.
            model_response: The model's generated response.
            judge_result: Optional LLM judge evaluation.

        Returns:
            EvalResult with scores and metadata.
        """
        ...

    def load_data(
        self,
        max_samples: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> List[BenchmarkItem]:
        """Load benchmark data from disk."""
        self._items = self.loader.load_benchmark(
            self.name,
            max_samples=max_samples,
            categories=categories,
        )
        return self._items

    @property
    def items(self) -> List[BenchmarkItem]:
        """Get loaded benchmark items."""
        if self._items is None:
            self.load_data()
        return self._items or []

    def get_few_shot_examples(self, num_examples: Optional[int] = None) -> str:
        """
        Generate few-shot example text for the prompt.

        Args:
            num_examples: Number of examples (default: self.num_few_shot)

        Returns:
            Formatted few-shot examples string.
        """
        n = num_examples or self.num_few_shot
        if n == 0 or not self.items:
            return ""

        examples = self.items[:n]
        example_texts = []
        for ex in examples:
            example_texts.append(
                f"Soru: {ex.question}\nCevap: {ex.correct_answer}"
            )
        return "\n\n".join(example_texts) + "\n\n"

    def compute_aggregate(self, results: List[EvalResult]) -> BenchmarkResults:
        """
        Compute aggregate metrics from individual results.

        Args:
            results: List of individual EvalResult objects.

        Returns:
            Aggregated BenchmarkResults.
        """
        if not results:
            return BenchmarkResults(
                benchmark_name=self.name,
                model_name="",
                score=0.0,
                num_samples=0,
                num_correct=0,
            )

        # Overall score
        total_score = sum(r.score for r in results)
        num_correct = sum(1 for r in results if r.is_correct)

        # Category scores
        category_scores: Dict[str, List[float]] = {}
        for r in results:
            if r.category not in category_scores:
                category_scores[r.category] = []
            category_scores[r.category].append(r.score)

        avg_category_scores = {
            cat: sum(scores) / len(scores)
            for cat, scores in category_scores.items()
        }

        return BenchmarkResults(
            benchmark_name=self.name,
            model_name=results[0].metadata.get("model_name", "") if results else "",
            score=total_score / len(results),
            num_samples=len(results),
            num_correct=num_correct,
            category_scores=avg_category_scores,
            individual_results=results,
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Get benchmark metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "num_items": len(self.items),
            "num_few_shot": self.num_few_shot,
            "categories": list(set(item.category for item in self.items)),
        }
