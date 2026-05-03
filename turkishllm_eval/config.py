"""
Global configuration management for TurkishLLM-Eval.

Handles YAML config loading, environment variable overrides,
model presets, and port assignments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

# ─── Project-specific port assignments (avoid conflicts) ───
# These ports are deliberately chosen to be uncommon and avoid
# clashing with common dev ports (3000, 4000, 5000, 8000, 8080, 8501, etc.)
PORTS = {
    "gradio_leaderboard": 7847,
    "mock_api_server": 7848,
    "metrics_dashboard": 7849,
}

# ─── Base paths ───
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / ".cache"


# ─── Model Registry: Turkish LLM presets ───
TURKISH_MODEL_REGISTRY = {
    "trendyol-7b": {
        "model_id": "Trendyol/Trendyol-LLM-7b-chat-v2.0",
        "type": "huggingface",
        "display_name": "Trendyol LLM 7B Chat",
        "parameters": "7B",
        "developer": "Trendyol",
    },
    "trendyol-8b-llama3": {
        "model_id": "Trendyol/Llama-3-Trendyol-LLM-8b-chat-v2.0",
        "type": "huggingface",
        "display_name": "Trendyol Llama-3 8B Chat",
        "parameters": "8B",
        "developer": "Trendyol",
    },
    "turkcell-7b": {
        "model_id": "Turkcell/Turkcell-LLM-7b-v1",
        "type": "huggingface",
        "display_name": "Turkcell LLM 7B",
        "parameters": "7B",
        "developer": "Turkcell",
    },
    "cosmos-turkish-llama": {
        "model_id": "ytu-ce/turkish-llama-8b-DPO-v0.1",
        "type": "huggingface",
        "display_name": "YTU-CE Turkish Llama 8B",
        "parameters": "8B",
        "developer": "YTU-CE Cosmos",
    },
    "cosmos-gemma-9b": {
        "model_id": "ytu-ce/Turkish-Gemma-2-9b-it-DPO",
        "type": "huggingface",
        "display_name": "YTU-CE Turkish Gemma 9B",
        "parameters": "9B",
        "developer": "YTU-CE Cosmos",
    },
    "kumru-7b": {
        "model_id": "VNGRS-AI/Kumru-7B",
        "type": "huggingface",
        "display_name": "Kumru 7B (VNGRS)",
        "parameters": "7.4B",
        "developer": "VNGRS",
    },
    "wiro-turkish-9b": {
        "model_id": "WiroAI/wiroai-turkish-llm-9b",
        "type": "huggingface",
        "display_name": "WiroAI Turkish 9B",
        "parameters": "9B",
        "developer": "WiroAI",
    },
    "gpt-4o": {
        "model_id": "gpt-4o",
        "type": "openai",
        "display_name": "GPT-4o (OpenAI)",
        "parameters": "~200B+",
        "developer": "OpenAI",
    },
    "gpt-4o-mini": {
        "model_id": "gpt-4o-mini",
        "type": "openai",
        "display_name": "GPT-4o Mini (OpenAI)",
        "parameters": "~8B",
        "developer": "OpenAI",
    },
    "claude-3.5-sonnet": {
        "model_id": "claude-3-5-sonnet-20241022",
        "type": "anthropic",
        "display_name": "Claude 3.5 Sonnet",
        "parameters": "~70B+",
        "developer": "Anthropic",
    },
}


@dataclass
class JudgeConfig:
    """Configuration for LLM-as-a-judge pipeline."""

    primary_judge: str = "gpt-4o"
    secondary_judge: str = "claude-3-5-sonnet-20241022"
    ensemble_strategy: str = "weighted_average"  # weighted_average, majority_vote, max_agreement
    primary_weight: float = 0.55
    secondary_weight: float = 0.45
    temperature: float = 0.0
    max_tokens: int = 1024
    cache_judgments: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    benchmarks: List[str] = field(
        default_factory=lambda: ["truthfulqa_tr", "mmlu_tr", "hallucination_tr", "bias_tr"]
    )
    num_few_shot: int = 0  # 0-shot by default
    max_samples: Optional[int] = None  # None = use all samples
    batch_size: int = 8
    generation_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": True,
        }
    )


@dataclass
class EvalConfig:
    """Master configuration for TurkishLLM-Eval."""

    # Model
    model_name: str = "trendyol-7b"
    model_id: Optional[str] = None  # Override registry lookup
    model_type: str = "huggingface"  # huggingface, openai, anthropic, ollama, vllm

    # Benchmarks
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Judge
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    # Infrastructure
    device: str = "auto"  # auto, cpu, cuda, mps
    dtype: str = "float16"  # float16, bfloat16, float32
    use_flash_attention: bool = True

    # Ports (project-specific, non-conflicting)
    gradio_port: int = PORTS["gradio_leaderboard"]  # 7847
    api_port: int = PORTS["mock_api_server"]  # 7848

    # Output
    output_dir: str = "results"
    save_individual_responses: bool = True
    export_format: str = "json"  # json, csv, both

    # API Keys (from env vars)
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    # Logging
    log_level: str = "INFO"
    verbose: bool = False

    def resolve_model(self) -> Dict[str, Any]:
        """Resolve model details from registry or direct specification."""
        if self.model_id:
            return {
                "model_id": self.model_id,
                "type": self.model_type,
                "display_name": self.model_id,
                "parameters": "Unknown",
                "developer": "Unknown",
            }

        if self.model_name in TURKISH_MODEL_REGISTRY:
            return TURKISH_MODEL_REGISTRY[self.model_name]

        return {
            "model_id": self.model_name,
            "type": self.model_type,
            "display_name": self.model_name,
            "parameters": "Unknown",
            "developer": "Unknown",
        }


def load_config(config_path: str | Path | None = None) -> EvalConfig:
    """
    Load evaluation configuration from YAML file with environment overrides.

    Args:
        config_path: Path to YAML config file. If None, uses defaults.

    Returns:
        EvalConfig instance with merged settings.
    """
    config = EvalConfig()

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

            # Apply top-level settings
            for key, value in yaml_data.items():
                if key == "benchmark" and isinstance(value, dict):
                    for bk, bv in value.items():
                        if hasattr(config.benchmark, bk):
                            setattr(config.benchmark, bk, bv)
                elif key == "judge" and isinstance(value, dict):
                    for jk, jv in value.items():
                        if hasattr(config.judge, jk):
                            setattr(config.judge, jk, jv)
                elif hasattr(config, key):
                    setattr(config, key, value)

    # Environment variable overrides
    env_overrides = {
        "TURKEVAL_MODEL": "model_name",
        "TURKEVAL_MODEL_ID": "model_id",
        "TURKEVAL_DEVICE": "device",
        "TURKEVAL_OUTPUT_DIR": "output_dir",
        "TURKEVAL_LOG_LEVEL": "log_level",
        "TURKEVAL_GRADIO_PORT": "gradio_port",
        "TURKEVAL_API_PORT": "api_port",
    }

    for env_key, config_key in env_overrides.items():
        env_val = os.getenv(env_key)
        if env_val is not None:
            # Handle int conversion for ports
            if "port" in config_key.lower():
                env_val = int(env_val)
            setattr(config, config_key, env_val)

    return config


def get_port(service: str) -> int:
    """Get the assigned port for a service. Raises KeyError if unknown service."""
    if service not in PORTS:
        raise KeyError(
            f"Unknown service '{service}'. Available: {list(PORTS.keys())}"
        )
    return PORTS[service]
