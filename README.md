<div align="center">

# 🇹🇷 TurkishLLM-Eval

### Comprehensive Hallucination, Factual Accuracy & Bias Benchmark Suite for Turkish LLMs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97-Leaderboard-yellow)](https://huggingface.co/spaces/saciducak/turkishllm-eval)

*The first open-source evaluation framework designed specifically for Turkish Large Language Models*

</div>

---

## 🎯 What is TurkishLLM-Eval?

TurkishLLM-Eval is a comprehensive benchmark suite that evaluates Turkish LLMs across four critical dimensions:

| Benchmark | Description | Focus |
|-----------|-------------|-------|
| **🎯 TruthfulQA-TR** | Turkish adaptation of TruthfulQA | Common misconceptions & factual accuracy |
| **📚 MMLU-TR** | Multi-domain knowledge test | Literature, History, Law, Geography, Culture |
| **🔍 Hallucination-TR** | Hallucination detection | Fabrication, entity confusion, temporal errors |
| **⚖️ Bias-TR** | Bias detection (Turkey-specific) | Gender, ethnic, sectarian, regional bias |

### Composite Score: TurkEval™

```
TurkEval™ = 0.30 × TruthfulQA-TR + 0.25 × MMLU-TR + 0.25 × Anti-Hallucination + 0.20 × Anti-Bias
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    EvalPipeline                          │
├──────────┬──────────┬───────────────┬──────────────────┤
│  Model   │Benchmarks│  Judge        │    Metrics        │
│ Adapters │          │  Ensemble     │                   │
├──────────┼──────────┼───────────────┼──────────────────┤
│ HF Trans │TruthfulQA│ GPT-4o Judge  │ TurkEval Score   │
│ OpenAI   │MMLU-TR   │ Claude Judge  │ Accuracy          │
│ Ollama   │Halluc-TR │ Weighted Avg  │ Cohen's Kappa     │
│ vLLM     │Bias-TR   │ Agreement     │ Bias Rate         │
└──────────┴──────────┴───────────────┴──────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/saciducak/turkishllm-eval.git
cd turkishllm-eval
pip install -e ".[dev]"
cp .env.example .env  # Add your API keys
```

### Run Evaluation

```bash
# List available model presets
python scripts/run_eval.py --list-models

# Quick evaluation (subset)
python scripts/run_eval.py --model trendyol-7b --config configs/quick_eval.yaml

# Full evaluation
python scripts/run_eval.py --model turkcell-7b --config configs/full_eval.yaml

# Specific benchmark only
python scripts/run_eval.py --model cosmos-turkish-llama --benchmark truthfulqa_tr mmlu_tr

# Direct HuggingFace model ID
python scripts/run_eval.py --model-id ytu-ce/turkish-llama-8b-DPO-v0.1 --model-type huggingface
```

### Launch Leaderboard (Port 7847)

```bash
python app/app.py
# Opens at http://localhost:7847
```

## 🏆 Supported Turkish LLMs

| Model | Developer | Parameters | Type |
|-------|-----------|------------|------|
| Trendyol-LLM-7b-chat | Trendyol | 7B | HuggingFace |
| Trendyol Llama-3 8B | Trendyol | 8B | HuggingFace |
| Turkcell-LLM-7b-v1 | Turkcell | 7B | HuggingFace |
| Turkish-Llama-8b-DPO | YTU-CE Cosmos | 8B | HuggingFace |
| Turkish-Gemma-9b | YTU-CE Cosmos | 9B | HuggingFace |
| Kumru-7B | VNGRS | 7.4B | HuggingFace |
| WiroAI Turkish 9B | WiroAI | 9B | HuggingFace |
| GPT-4o | OpenAI | ~200B+ | API |
| Claude 3.5 Sonnet | Anthropic | ~70B+ | API |

## 🔬 Judge Pipeline

TurkishLLM-Eval uses an **LLM-as-a-judge ensemble**:

1. **GPT-4o** (primary, weight: 0.55) — evaluates truthfulness, hallucination severity, bias
2. **Claude 3.5 Sonnet** (secondary, weight: 0.45) — cross-validates with independent scoring
3. **Ensemble aggregation** — weighted average with inter-judge agreement (Cohen's κ)

## 📊 Bias Taxonomy (Turkey-Specific)

| Category | Turkish | Examples |
|----------|---------|----------|
| Gender | Cinsiyet | Professional stereotypes, family roles |
| Ethnic | Etnik | Kurdish, Arab, Laz, Circassian stereotypes |
| Sectarian | Mezhepsel | Sunni/Alevi discrimination |
| Regional | Bölgesel | East-West, urban-rural divide |
| Socioeconomic | Sosyoekonomik | Class, education level stereotypes |

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📁 Project Structure

```
turkishllm-eval/
├── turkishllm_eval/          # Core Python package
│   ├── benchmarks/           # Benchmark implementations
│   ├── judges/               # LLM-as-a-judge pipeline
│   ├── models/               # Model adapters
│   ├── metrics/              # Evaluation metrics
│   ├── data/                 # Data loading
│   ├── utils/                # Utilities
│   └── pipeline.py           # Main eval pipeline
├── data/                     # Benchmark datasets (JSONL)
├── configs/                  # YAML configurations
├── app/                      # Gradio leaderboard (port 7847)
├── scripts/                  # CLI tools
├── tests/                    # Test suite
└── docs/                     # Documentation
```

## 🔧 Configuration

All configs use **project-specific ports** to avoid conflicts:
- **Gradio Leaderboard:** `7847`
- **Mock API Server:** `7848`
- **Metrics Dashboard:** `7849`

## 📄 License

Apache License 2.0 — See [LICENSE](LICENSE) for details.

## 👤 Author

**Muhammed Sacid Ucak** — AI/NLP Engineer  
[GitHub](https://github.com/saciducak) • [LinkedIn](https://linkedin.com/in/saciducak)

---

<div align="center">
<i>Built with ❤️ for the Turkish AI community</i>
</div>
