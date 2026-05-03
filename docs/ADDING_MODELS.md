# Adding New Models to TurkishLLM-Eval

## Quick Start: Model Presets

The fastest way is to add your model to the preset registry in `turkishllm_eval/config.py`:

```python
TURKISH_MODEL_REGISTRY = {
    # Add your model here:
    "my-model": {
        "model_id": "organization/model-name-on-huggingface",
        "type": "huggingface",  # huggingface, openai, ollama, vllm
        "display_name": "My Model 7B",
        "parameters": "7B",
        "developer": "Your Organization",
    },
}
```

Then run:
```bash
python scripts/run_eval.py --model my-model
```

## Without Registry: Direct Model ID

```bash
python scripts/run_eval.py --model-id organization/model-name --model-type huggingface
```

## Supported Model Types

| Type | Adapter | Use Case |
|------|---------|----------|
| `huggingface` | `HuggingFaceModel` | Models on HuggingFace Hub |
| `openai` | `OpenAIModel` | OpenAI API models (GPT-4o, etc.) |
| `ollama` | `OllamaModel` | Locally running Ollama models |
| `vllm` | `VLLMModel` | High-throughput inference with vLLM |

## Custom Model Adapter

To add a completely new model backend, extend `BaseModel`:

```python
from turkishllm_eval.models.base import BaseModel, ModelResponse

class MyCustomModel(BaseModel):
    def generate(self, prompt, system_prompt="", max_new_tokens=512, 
                 temperature=0.1, top_p=0.9, **kwargs) -> ModelResponse:
        # Your inference logic here
        text = my_inference(prompt)
        return ModelResponse(text=text, model_id=self.model_id)
    
    def batch_generate(self, prompts, **kwargs) -> list[ModelResponse]:
        return [self.generate(p, **kwargs) for p in prompts]
    
    def is_available(self) -> bool:
        return True
```

Register it in `turkishllm_eval/pipeline.py` model initialization.
