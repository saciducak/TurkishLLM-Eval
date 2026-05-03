"""
Structured logging for TurkishLLM-Eval with Rich formatting.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for TurkishLLM-Eval
TURKEVAL_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "benchmark": "bold magenta",
        "model": "bold blue",
        "score": "bold yellow",
        "judge": "bold cyan",
    }
)

console = Console(theme=TURKEVAL_THEME)


def setup_logger(
    name: str = "turkishllm_eval",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a structured logger with Rich formatting.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


def log_benchmark_start(logger: logging.Logger, benchmark_name: str, model_name: str):
    """Log the start of a benchmark evaluation."""
    logger.info(
        f"[benchmark]━━━ Starting benchmark: {benchmark_name} [/benchmark] "
        f"│ [model]Model: {model_name}[/model]"
    )


def log_benchmark_result(
    logger: logging.Logger,
    benchmark_name: str,
    score: float,
    num_samples: int,
):
    """Log benchmark result with formatting."""
    emoji = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
    logger.info(
        f"{emoji} [benchmark]{benchmark_name}[/benchmark] │ "
        f"[score]Score: {score:.4f}[/score] │ "
        f"Samples: {num_samples}"
    )


def log_judge_verdict(
    logger: logging.Logger,
    judge_name: str,
    question_id: str,
    score: float,
    reasoning: str = "",
):
    """Log a judge's verdict on a specific question."""
    logger.debug(
        f"[judge]{judge_name}[/judge] │ Q:{question_id} │ "
        f"[score]{score:.2f}[/score]"
        + (f" │ {reasoning[:80]}..." if reasoning else "")
    )


# Module-level logger
logger = setup_logger()
