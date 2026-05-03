#!/usr/bin/env python3
"""
Submit evaluation results to the TurkishLLM-Eval leaderboard.

Usage:
    python scripts/submit_to_leaderboard.py results/eval_model_timestamp.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()

LEADERBOARD_FILE = Path(__file__).parent.parent / "app" / "sample_results.json"


def submit(results_file: Path) -> None:
    """Add results to the leaderboard data file."""
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Extract leaderboard entry
    entry = {
        "model_name": results.get("display_name", results.get("model_name", "Unknown")),
        "developer": results.get("developer", "Unknown"),
        "parameters": results.get("parameters", "Unknown"),
        "turkeval_score": round(results.get("turkeval_score", 0), 1),
    }

    # Add per-benchmark scores
    for bname, bdata in results.get("benchmarks", {}).items():
        if isinstance(bdata, dict):
            entry[bname] = round(bdata.get("score", 0) * 100, 1)

    # Load existing leaderboard
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
            leaderboard = json.load(f)
    else:
        leaderboard = []

    # Check for duplicate
    existing = [i for i, e in enumerate(leaderboard) if e["model_name"] == entry["model_name"]]
    if existing:
        console.print(f"[yellow]Updating existing entry for {entry['model_name']}[/yellow]")
        leaderboard[existing[0]] = entry
    else:
        leaderboard.append(entry)

    # Sort by TurkEval score
    leaderboard.sort(key=lambda x: x.get("turkeval_score", 0), reverse=True)

    # Save
    with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    console.print(f"[green]✓ Submitted {entry['model_name']} to leaderboard[/green]")
    console.print(f"  TurkEval™ Score: {entry['turkeval_score']}")
    console.print(f"  Leaderboard: {LEADERBOARD_FILE}")

    # Print current standings
    table = Table(title="🏆 Current Leaderboard")
    table.add_column("Rank", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("TurkEval™", style="green")
    for i, e in enumerate(leaderboard, 1):
        style = "bold green" if e["model_name"] == entry["model_name"] else ""
        table.add_row(str(i), e["model_name"], f"{e['turkeval_score']:.1f}", style=style)
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Submit to leaderboard")
    parser.add_argument("results_file", type=str, help="Results JSON file path")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        console.print(f"[red]File not found: {results_path}[/red]")
        sys.exit(1)

    submit(results_path)


if __name__ == "__main__":
    main()
