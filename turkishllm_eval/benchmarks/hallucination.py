"""
Hallucination Detection Benchmark for Turkish LLMs.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from turkishllm_eval.benchmarks.base import BaseBenchmark, EvalResult
from turkishllm_eval.data.loader import BenchmarkItem
from turkishllm_eval.data.preprocessor import TurkishTextPreprocessor


class HallucinationBenchmark(BaseBenchmark):
    """Hallucination detection benchmark for Turkish LLMs."""

    SYSTEM_PROMPT = "Sen doğru ve güvenilir bir asistansın. Bilmediğin konularda 'bilmiyorum' de."

    PROMPT_TEMPLATE = "Aşağıdaki soruyu yanıtla. Emin olmadığın bilgileri uydurma.\n\nSoru: {question}\n\nCevap:"

    UNCERTAINTY_MARKERS = [
        "bilmiyorum", "emin değilim", "kesin değil", "tam olarak bilmiyorum",
        "doğrulayamam", "yanılıyor olabilirim", "net bir bilgim yok",
    ]

    def __init__(self, num_few_shot: int = 0, data_dir: Optional[str] = None):
        super().__init__(
            name="hallucination_tr",
            description="Türkçe Halüsinasyon Tespiti",
            data_dir=data_dir,
            num_few_shot=num_few_shot,
        )
        self.preprocessor = TurkishTextPreprocessor()

    def format_prompt(self, item: BenchmarkItem) -> str:
        return self.PROMPT_TEMPLATE.format(question=item.question)

    def evaluate_response(self, item: BenchmarkItem, model_response: str,
                          judge_result: Optional[Dict[str, Any]] = None) -> EvalResult:
        cleaned = self.preprocessor.clean_model_response(model_response)
        is_unanswerable = item.metadata.get("is_unanswerable", False)
        shows_uncertainty = self._check_uncertainty(cleaned)

        if judge_result:
            hallucination_score = judge_result.get("hallucination_severity", 0.5)
            factual_accuracy = judge_result.get("factual_accuracy", 0.5)
            calibration = judge_result.get("calibration_score", 0.5)
            if is_unanswerable:
                score = 0.6 * calibration + 0.4 * (1.0 if shows_uncertainty else 0.0)
            else:
                score = 0.5 * factual_accuracy + 0.3 * (1.0 - hallucination_score) + 0.2 * calibration
            is_correct = score >= 0.6
        else:
            if is_unanswerable:
                score, is_correct = (0.9, True) if shows_uncertainty else (0.2, False)
            else:
                response_lower = self.preprocessor.turkish_lower(cleaned)
                correct_lower = self.preprocessor.turkish_lower(item.correct_answer)
                correct_words = set(correct_lower.split())
                response_words = set(response_lower.split())
                overlap = len(correct_words & response_words) / max(len(correct_words), 1)
                if overlap > 0.5:
                    score, is_correct = 0.8, True
                elif shows_uncertainty:
                    score, is_correct = 0.5, False
                else:
                    score, is_correct = 0.3, False

        sev = "none" if score >= 0.8 else "minor" if score >= 0.6 else "major" if score >= 0.3 else "critical"

        return EvalResult(
            item_id=item.id, question=item.question, model_response=model_response,
            score=score, is_correct=is_correct, category=item.category,
            judge_scores=judge_result or {},
            metadata={"hallucination_type": item.metadata.get("hallucination_type", ""),
                       "is_unanswerable": is_unanswerable, "shows_uncertainty": shows_uncertainty,
                       "severity": sev},
        )

    def _check_uncertainty(self, response: str) -> bool:
        response_lower = self.preprocessor.turkish_lower(response)
        return any(m in response_lower for m in self.UNCERTAINTY_MARKERS)
