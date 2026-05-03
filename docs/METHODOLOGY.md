# TurkishLLM-Eval — Evaluation Methodology

## Overview

TurkishLLM-Eval evaluates Turkish Large Language Models across four critical dimensions using a multi-judge ensemble pipeline with culturally-adapted benchmarks.

## Benchmark Design Principles

### 1. Cultural Adaptation Over Translation
All benchmarks are **culturally adapted** for the Turkish context, not simply translated from English equivalents. This means:
- Questions reference Turkish historical events, geography, literature
- Bias categories reflect Turkish societal dynamics
- Misconceptions are specific to Turkish cultural context
- Language patterns consider Turkish morphology (agglutinative structure, vowel harmony)

### 2. Multi-Dimensional Evaluation
Rather than a single score, models are evaluated across orthogonal dimensions:

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Truthfulness | 30% | Resistance to reproducing common falsehoods |
| Factual Accuracy | 25% | Correct answers on knowledge questions |
| Anti-Hallucination | 25% | Tendency to fabricate information |
| Anti-Bias | 20% | Fairness across demographic groups |

### 3. Composite TurkEval™ Score
```
TurkEval™ = 0.30 × TruthfulQA-TR + 0.25 × MMLU-TR + 0.25 × Hallucination-TR + 0.20 × Bias-TR
```

Grading scale:
- **A+ (≥90):** Exceptional — production-ready for Turkish
- **A (≥80):** Strong — reliable with minor edge cases
- **B (≥70):** Good — usable with supervision
- **C (≥60):** Fair — significant limitations
- **D (≥50):** Poor — unreliable for Turkish
- **F (<50):** Failing — not recommended

---

## Benchmark Details

### TruthfulQA-TR (30% weight)
**Goal:** Test whether models reproduce common Turkish misconceptions.

**Design:** Adapted from [Lin et al., 2022](https://arxiv.org/abs/2109.07958) with Turkish-specific questions across 7 categories: History (Tarih), Geography (Coğrafya), Science (Bilim), Health (Sağlık), Culture (Kültür), Economy (Ekonomi), Language (Dil).

**Evaluation modes:**
- **MC (Multiple-Choice):** 4-option questions, strict accuracy
- **Generation:** Open-ended, evaluated by judge ensemble

**Scoring:** MC accuracy + judge truthfulness score (blended 60/40)

### MMLU-TR (25% weight)
**Goal:** Measure broad factual knowledge in Turkish-specific domains.

**Design:** 4-choice MC format inspired by [Hendrycks et al., 2020](https://arxiv.org/abs/2009.03300), adapted to 5 Turkish domains:
1. **Türk Edebiyatı** — Literary works, authors, periods
2. **Türkiye Tarihi** — Ottoman, Republic, modern history
3. **Türk Hukuku** — Constitutional law, legal system basics
4. **Türkiye Coğrafyası** — Physical/political geography
5. **Türk Kültürü** — Traditions, arts, social sciences

**Scoring:** Strict accuracy (exact match on correct choice)

### Hallucination-TR (25% weight)
**Goal:** Detect tendency to fabricate information.

**Hallucination types tested:**
1. **Factual Fabrication** — Inventing non-existent events/people
2. **Entity Confusion** — Mixing up similar names/places
3. **Temporal Hallucination** — Date/chronology errors
4. **Attribution Error** — Misattributing quotes/works
5. **Overconfident Speculation** — Presenting guesses as facts

**Key metrics:**
- **Hallucination Rate:** % of responses with hallucinated content
- **Severity:** none / minor / major / critical
- **Calibration:** Appropriate use of "bilmiyorum" (I don't know)
- **Refusal Accuracy:** Correct refusal on unanswerable questions

### Bias-TR (20% weight)
**Goal:** Detect systematic biases in Turkey-specific context.

**Categories:** See [BIAS_TAXONOMY.md](BIAS_TAXONOMY.md) for full taxonomy.

**Testing approach:**
- **Paired prompts** (stereotypical vs. anti-stereotypical)
- **BBQ-style** ambiguous context testing
- **Sentiment analysis** on responses about different groups
- **Stereotype detection** via keyword + judge analysis

---

## Judge Pipeline

### Architecture
```
Model Response → GPT-4o Judge (w=0.55) ─┐
                                          ├─→ Weighted Average → Final Score
Model Response → Claude Judge (w=0.45) ──┘
                                          └─→ Inter-Judge Agreement (Cohen's κ)
```

### Why Ensemble?
- Single judges have systematic biases
- Ensemble reduces variance and increases reliability
- Inter-judge agreement provides a confidence signal
- Disagreements flag items needing human review

### Rubrics
All rubrics are written in Turkish to ensure judges evaluate in the target language context. Each benchmark type has a specific rubric with structured JSON output format.

### Quality Assurance
- **Cohen's Kappa (κ):** Measures inter-judge agreement
  - κ > 0.80: Almost perfect agreement
  - κ 0.60-0.80: Substantial agreement
  - κ 0.40-0.60: Moderate agreement (minimum acceptable)
  - κ < 0.40: Fair/poor — results should be flagged

---

## Limitations

1. **Dataset size:** Initial release has 80+ questions; production benchmarks typically have 500+
2. **Judge bias:** LLM judges may have their own biases toward certain response styles
3. **Cultural subjectivity:** Some bias evaluations involve subjective judgment
4. **Cost:** Full evaluation with judge ensemble requires API credits
5. **Temporal drift:** Knowledge-based questions may become outdated

## References

- Lin et al. (2022). "TruthfulQA: Measuring How Models Mimic Human Falsehoods"
- Hendrycks et al. (2020). "Measuring Massive Multitask Language Understanding"
- Parrish et al. (2022). "BBQ: A Hand-Built Bias Benchmark for Question Answering"
- Zheng et al. (2024). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
