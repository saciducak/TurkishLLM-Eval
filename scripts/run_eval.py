#!/usr/bin/env python3
"""
TurkishLLM-Eval — Main Evaluation Runner CLI.

Usage:
    python scripts/run_eval.py --model trendyol-7b --config configs/default.yaml
    python scripts/run_eval.py --model-id Turkcell/Turkcell-LLM-7b-v1 --benchmark truthfulqa_tr
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from turkishllm_eval.config import load_config, TURKISH_MODEL_REGISTRY
from turkishllm_eval.pipeline import EvalPipeline
from turkishllm_eval.utils.logger import logger, console
from rich.table import Table


def main():
    parser = argparse.ArgumentParser(description="TurkishLLM-Eval Runner")
    parser.add_argument("--model", type=str, default="trendyol-7b", help="Model preset name")
    parser.add_argument("--model-id", type=str, default=None, help="Direct HuggingFace model ID")
    parser.add_argument("--model-type", type=str, default="huggingface", choices=["huggingface", "openai", "ollama", "vllm"])
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--benchmark", type=str, nargs="+", default=None, help="Specific benchmarks to run")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per benchmark")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--list-models", action="store_true", help="List available model presets")
    args = parser.parse_args()

    if args.list_models:
        table = Table(title="🇹🇷 Available Turkish LLM Presets")
        table.add_column("Preset", style="cyan")
        table.add_column("Model ID", style="green")
        table.add_column("Developer", style="yellow")
        table.add_column("Parameters", style="magenta")
        for name, info in TURKISH_MODEL_REGISTRY.items():
            table.add_row(name, info["model_id"], info["developer"], info["parameters"])
        console.print(table)
        return

    config = load_config(args.config)
    if args.model_id:
        config.model_id = args.model_id
        config.model_type = args.model_type
    else:
        config.model_name = args.model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_samples:
        config.benchmark.max_samples = args.max_samples

    pipeline = EvalPipeline(config)
    results = pipeline.run(benchmarks=args.benchmark, max_samples=args.max_samples)

    # Summary table
    table = Table(title=f"🏆 Results: {results.get('display_name', args.model)}")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Accuracy", style="yellow")
    for bname, bdata in results.get("benchmarks", {}).items():
        table.add_row(bname, f"{bdata['score']:.4f}", f"{bdata.get('accuracy_pct', 0):.1f}%")
    table.add_row("━" * 20, "━" * 10, "━" * 10)
    table.add_row("TurkEval™ Score", f"{results.get('turkeval_score', 0):.1f}", results.get("turkeval_grade", ""))
    console.print(table)


if __name__ == "__main__":
    main()
