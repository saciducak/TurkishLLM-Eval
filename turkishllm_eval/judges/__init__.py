"""LLM-as-a-Judge subpackage."""

from turkishllm_eval.judges.base import BaseJudge, JudgmentResult
from turkishllm_eval.judges.gpt4_judge import GPT4Judge
from turkishllm_eval.judges.claude_judge import ClaudeJudge
from turkishllm_eval.judges.ensemble import JudgeEnsemble
from turkishllm_eval.judges.rubrics import TurkishEvalRubrics

__all__ = [
    "BaseJudge", "JudgmentResult",
    "GPT4Judge", "ClaudeJudge",
    "JudgeEnsemble", "TurkishEvalRubrics",
]
