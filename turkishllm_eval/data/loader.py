"""
Dataset loader for TurkishLLM-Eval benchmark data.

Loads JSONL format benchmark questions with metadata, supports
filtering, sampling, and split management.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from turkishllm_eval.config import DATA_DIR
from turkishllm_eval.utils.logger import logger


@dataclass
class BenchmarkItem:
    """A single benchmark question/test case."""

    id: str
    question: str
    category: str
    correct_answer: str
    incorrect_answers: List[str] = field(default_factory=list)
    choices: List[str] = field(default_factory=list)
    correct_choice_index: int = -1
    reference_context: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)
    benchmark_type: str = ""  # truthfulqa, mmlu, hallucination, bias

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "correct_answer": self.correct_answer,
            "incorrect_answers": self.incorrect_answers,
            "choices": self.choices,
            "correct_choice_index": self.correct_choice_index,
            "reference_context": self.reference_context,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "benchmark_type": self.benchmark_type,
        }


class DatasetLoader:
    """
    Loads benchmark datasets from JSONL files.

    Supports:
    - Loading from local data/ directory
    - Category-based filtering
    - Random sampling with seed
    - Dataset statistics
    """

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def load_benchmark(
        self,
        benchmark_name: str,
        max_samples: Optional[int] = None,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        seed: int = 42,
    ) -> List[BenchmarkItem]:
        """
        Load a benchmark dataset.

        Args:
            benchmark_name: Name of the benchmark (e.g., 'truthfulqa_tr')
            max_samples: Maximum number of samples to load
            categories: Filter by categories
            difficulties: Filter by difficulty levels
            seed: Random seed for sampling

        Returns:
            List of BenchmarkItem instances
        """
        benchmark_dir = self.data_dir / benchmark_name
        questions_file = benchmark_dir / "questions.jsonl"

        if not questions_file.exists():
            logger.warning(f"Dataset file not found: {questions_file}")
            return []

        items = []
        with open(questions_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    item = BenchmarkItem(
                        id=data.get("id", f"{benchmark_name}_{line_num}"),
                        question=data["question"],
                        category=data.get("category", "general"),
                        correct_answer=data.get("correct_answer", ""),
                        incorrect_answers=data.get("incorrect_answers", []),
                        choices=data.get("choices", []),
                        correct_choice_index=data.get("correct_choice_index", -1),
                        reference_context=data.get("reference_context", ""),
                        difficulty=data.get("difficulty", "medium"),
                        metadata=data.get("metadata", {}),
                        benchmark_type=benchmark_name,
                    )
                    items.append(item)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed line {line_num} in {questions_file}: {e}")

        # Apply filters
        if categories:
            items = [item for item in items if item.category in categories]

        if difficulties:
            items = [item for item in items if item.difficulty in difficulties]

        # Sample if needed
        if max_samples and len(items) > max_samples:
            import random
            rng = random.Random(seed)
            items = rng.sample(items, max_samples)

        logger.info(
            f"Loaded {len(items)} items from {benchmark_name}"
            + (f" (categories: {categories})" if categories else "")
        )
        return items

    def get_benchmark_stats(self, benchmark_name: str) -> Dict[str, Any]:
        """Get statistics about a benchmark dataset."""
        items = self.load_benchmark(benchmark_name)

        if not items:
            return {"total": 0, "categories": {}, "difficulties": {}}

        categories = {}
        difficulties = {}
        for item in items:
            categories[item.category] = categories.get(item.category, 0) + 1
            difficulties[item.difficulty] = difficulties.get(item.difficulty, 0) + 1

        return {
            "total": len(items),
            "categories": categories,
            "difficulties": difficulties,
            "benchmark_name": benchmark_name,
        }

    def list_available_benchmarks(self) -> List[str]:
        """List all available benchmark datasets."""
        benchmarks = []
        if self.data_dir.exists():
            for subdir in sorted(self.data_dir.iterdir()):
                if subdir.is_dir() and (subdir / "questions.jsonl").exists():
                    benchmarks.append(subdir.name)
        return benchmarks
