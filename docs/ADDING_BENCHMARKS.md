# Adding New Benchmarks to TurkishLLM-Eval

## Step 1: Create Dataset

Create a JSONL file at `data/<benchmark_name>/questions.jsonl`:

```jsonl
{"id": "q_001", "question": "Your question?", "category": "cat1", "correct_answer": "Answer", "choices": ["A. Opt1", "B. Opt2", "C. Opt3", "D. Opt4"], "correct_choice_index": 0, "difficulty": "medium", "metadata": {}}
```

Add a `metadata.json`:
```json
{"name": "My Benchmark", "num_questions": 50, "language": "tr"}
```

## Step 2: Implement Benchmark Class

Create `turkishllm_eval/benchmarks/my_benchmark.py`:

```python
from turkishllm_eval.benchmarks.base import BaseBenchmark, EvalResult
from turkishllm_eval.data.loader import BenchmarkItem

class MyBenchmark(BaseBenchmark):
    SYSTEM_PROMPT = "System prompt for the model"
    
    def __init__(self, num_few_shot=0, data_dir=None):
        super().__init__(
            name="my_benchmark",  # Must match data/ folder name
            description="What this benchmark tests",
            data_dir=data_dir,
            num_few_shot=num_few_shot,
        )
    
    def format_prompt(self, item: BenchmarkItem) -> str:
        return f"Soru: {item.question}\nCevap:"
    
    def evaluate_response(self, item, model_response, judge_result=None) -> EvalResult:
        # Your evaluation logic
        is_correct = "expected" in model_response.lower()
        return EvalResult(
            item_id=item.id, question=item.question,
            model_response=model_response, score=1.0 if is_correct else 0.0,
            is_correct=is_correct, category=item.category,
        )
```

## Step 3: Register

Add to `turkishllm_eval/benchmarks/__init__.py` and `turkishllm_eval/pipeline.py`:

```python
BENCHMARK_REGISTRY = {
    # ...existing benchmarks...
    "my_benchmark": MyBenchmark,
}
```

## Step 4: Add to Config

In your YAML config:
```yaml
benchmark:
  benchmarks:
    - truthfulqa_tr
    - my_benchmark  # Add here
```

## Step 5: Update TurkEval Weights (Optional)

If adding to composite score, update `turkishllm_eval/metrics/composite.py`.
