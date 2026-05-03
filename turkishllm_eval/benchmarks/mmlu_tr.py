"""
Turkish MMLU Benchmark Implementation.

Evaluates Turkish LLMs on multi-domain knowledge with
4-choice multiple-choice questions across Turkish-specific
subject areas.

Categories:
- Türk Edebiyatı (Turkish Literature)
- Türkiye Tarihi (Turkish History)
- Türk Hukuku (Turkish Law)
- Türkiye Coğrafyası (Turkish Geography)
- Türk Kültürü & Sosyal Bilimler (Culture & Social Sciences)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from turkishllm_eval.benchmarks.base import BaseBenchmark, EvalResult
from turkishllm_eval.data.loader import BenchmarkItem
from turkishllm_eval.data.preprocessor import TurkishTextPreprocessor


class MMLUTurkish(BaseBenchmark):
    """
    Turkish MMLU (Massive Multitask Language Understanding) benchmark.

    Tests broad knowledge across Turkish-specific domains with
    standardized 4-choice multiple-choice format.
    """

    SYSTEM_PROMPT = (
        "Sen çok alanlı bilgi testlerinde uzmanlaşmış bir asistansın. "
        "Soruları dikkatle oku ve en doğru seçeneği belirle."
    )

    PROMPT_TEMPLATE = """Aşağıda {category} alanından bir soru verilmiştir. Doğru seçeneği belirle.

{few_shot}Soru: {question}

{choices}

Cevap (sadece harf):"""

    CATEGORY_NAMES = {
        "edebiyat": "Türk Edebiyatı",
        "tarih": "Türkiye Tarihi",
        "hukuk": "Türk Hukuku",
        "cografya": "Türkiye Coğrafyası",
        "kultur": "Türk Kültürü ve Sosyal Bilimler",
    }

    def __init__(self, num_few_shot: int = 0, data_dir: Optional[str] = None):
        super().__init__(
            name="mmlu_tr",
            description="Türkçe MMLU — Çok alanlı bilgi testi",
            data_dir=data_dir,
            num_few_shot=num_few_shot,
        )
        self.preprocessor = TurkishTextPreprocessor()

    def format_prompt(self, item: BenchmarkItem) -> str:
        """Format MMLU-style prompt."""
        few_shot = self.get_few_shot_examples()
        choices_text = self.preprocessor.format_choices(item.choices)

        category_display = self.CATEGORY_NAMES.get(item.category, item.category)

        return self.PROMPT_TEMPLATE.format(
            category=category_display,
            few_shot=few_shot,
            question=item.question,
            choices=choices_text,
        )

    def evaluate_response(
        self,
        item: BenchmarkItem,
        model_response: str,
        judge_result: Optional[Dict[str, Any]] = None,
    ) -> EvalResult:
        """Evaluate MC response — strict accuracy."""
        cleaned = self.preprocessor.clean_model_response(model_response)
        selected = self.preprocessor.extract_choice(cleaned, len(item.choices))

        is_correct = selected == item.correct_choice_index
        score = 1.0 if is_correct else 0.0

        return EvalResult(
            item_id=item.id,
            question=item.question,
            model_response=model_response,
            score=score,
            is_correct=is_correct,
            category=item.category,
            metadata={
                "selected_choice": selected,
                "correct_choice": item.correct_choice_index,
                "category_display": self.CATEGORY_NAMES.get(item.category, item.category),
            },
        )

    def get_category_report(self, results: List[EvalResult]) -> Dict[str, Dict[str, Any]]:
        """Generate per-category accuracy report."""
        category_data: Dict[str, Dict[str, Any]] = {}

        for r in results:
            cat = r.category
            if cat not in category_data:
                category_data[cat] = {"correct": 0, "total": 0, "display_name": ""}

            category_data[cat]["total"] += 1
            if r.is_correct:
                category_data[cat]["correct"] += 1
            category_data[cat]["display_name"] = self.CATEGORY_NAMES.get(cat, cat)

        # Compute accuracies
        for cat in category_data:
            total = category_data[cat]["total"]
            correct = category_data[cat]["correct"]
            category_data[cat]["accuracy"] = correct / total if total > 0 else 0.0

        return category_data
