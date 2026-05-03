"""
Result export utilities for TurkishLLM-Eval.

Supports JSON, CSV, and Hugging Face Dataset export formats.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from turkishllm_eval.utils.logger import logger


class ResultExporter:
    """Export evaluation results in multiple formats."""

    def __init__(self, output_dir: str | Path = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export results as a structured JSON file.

        Args:
            results: Evaluation results dictionary.
            filename: Custom filename. Auto-generated if None.

        Returns:
            Path to the exported file.
        """
        if filename is None:
            model_name = results.get("model_name", "unknown").replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{model_name}_{timestamp}.json"

        filepath = self.output_dir / filename

        # Add export metadata
        export_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "framework": "turkishllm-eval",
                "version": "0.1.0",
            },
            **self._serialize(results),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Results exported to: {filepath}")
        return filepath

    def export_csv(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export results as a CSV file (flattened benchmark scores).

        Args:
            results: Evaluation results dictionary.
            filename: Custom filename.

        Returns:
            Path to the exported file.
        """
        if filename is None:
            model_name = results.get("model_name", "unknown").replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_{model_name}_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Flatten results for CSV
        rows = self._flatten_results(results)

        if rows:
            with open(filepath, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"CSV exported to: {filepath}")
        return filepath

    def export_leaderboard_entry(
        self,
        results: Dict[str, Any],
        filename: str = "leaderboard_entry.json",
    ) -> Path:
        """
        Export a leaderboard submission entry.

        Args:
            results: Full evaluation results.
            filename: Output filename.

        Returns:
            Path to the entry file.
        """
        entry = {
            "model_name": results.get("model_name", "unknown"),
            "model_id": results.get("model_id", "unknown"),
            "display_name": results.get("display_name", results.get("model_name", "unknown")),
            "developer": results.get("developer", "Unknown"),
            "parameters": results.get("parameters", "Unknown"),
            "submitted_at": datetime.now().isoformat(),
            "scores": {},
        }

        # Extract benchmark scores
        for benchmark_name, benchmark_results in results.get("benchmarks", {}).items():
            if isinstance(benchmark_results, dict):
                entry["scores"][benchmark_name] = benchmark_results.get("score", 0.0)

        # Compute composite TurkEval score
        entry["turkeval_score"] = results.get("turkeval_score", 0.0)

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        logger.info(f"Leaderboard entry exported to: {filepath}")
        return filepath

    def _serialize(self, obj: Any) -> Any:
        """Recursively serialize objects for JSON export."""
        if hasattr(obj, "__dict__"):
            return {k: self._serialize(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def _flatten_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested results into CSV-friendly rows."""
        rows = []
        model_name = results.get("model_name", "unknown")

        for benchmark_name, benchmark_data in results.get("benchmarks", {}).items():
            if isinstance(benchmark_data, dict):
                row = {
                    "model_name": model_name,
                    "benchmark": benchmark_name,
                    "score": benchmark_data.get("score", 0.0),
                    "num_samples": benchmark_data.get("num_samples", 0),
                    "timestamp": results.get("timestamp", ""),
                }

                # Add category scores if available
                for cat_name, cat_score in benchmark_data.get("category_scores", {}).items():
                    row[f"category_{cat_name}"] = cat_score

                rows.append(row)

        return rows
