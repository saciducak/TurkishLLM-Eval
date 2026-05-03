"""
Turkish text preprocessor for TurkishLLM-Eval.

Handles Turkish-specific text normalization, character handling,
and prompt formatting for evaluation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional


class TurkishTextPreprocessor:
    """
    Preprocessor for Turkish text with special handling for:
    - Turkish-specific characters (ç, ğ, ı, ö, ş, ü, İ)
    - Whitespace normalization
    - Prompt template formatting
    """

    # Turkish-specific character mappings
    TURKISH_LOWER = "abcçdefgğhıijklmnoöprsştuüvyz"
    TURKISH_UPPER = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize Turkish text for comparison.

        - NFC unicode normalization
        - Whitespace collapsing
        - Removing zero-width characters
        """
        # Unicode normalization (NFC)
        text = unicodedata.normalize("NFC", text)

        # Remove zero-width characters
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def turkish_lower(text: str) -> str:
        """
        Turkish-aware lowercase conversion.
        Handles I→ı and İ→i correctly (unlike Python's default).
        """
        result = []
        for char in text:
            if char == "I":
                result.append("ı")
            elif char == "İ":
                result.append("i")
            else:
                result.append(char.lower())
        return "".join(result)

    @staticmethod
    def turkish_upper(text: str) -> str:
        """
        Turkish-aware uppercase conversion.
        Handles i→İ and ı→I correctly.
        """
        result = []
        for char in text:
            if char == "i":
                result.append("İ")
            elif char == "ı":
                result.append("I")
            else:
                result.append(char.upper())
        return "".join(result)

    @staticmethod
    def clean_model_response(response: str) -> str:
        """
        Clean model response for evaluation.

        Removes common artifacts like special tokens, excessive
        newlines, and formatting markers.
        """
        # Remove common special tokens
        special_tokens = [
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "</s>", "<s>", "[INST]", "[/INST]",
            "<<SYS>>", "<</SYS>>", "<|assistant|>",
            "<|user|>", "<|system|>",
        ]
        for token in special_tokens:
            response = response.replace(token, "")

        # Remove excessive newlines
        response = re.sub(r"\n{3,}", "\n\n", response)

        # Strip
        response = response.strip()

        return response

    @staticmethod
    def extract_choice(response: str, num_choices: int = 4) -> Optional[int]:
        """
        Extract multiple-choice answer from model response.

        Supports formats:
        - "A", "B", "C", "D"
        - "(A)", "(B)", etc.
        - "Cevap: A", "Yanıt: B"
        - "1", "2", "3", "4"

        Returns:
            Choice index (0-based) or None if extraction fails.
        """
        response = response.strip()
        labels = "ABCDEFGHIJ"[:num_choices]

        # Direct letter match (first occurrence)
        patterns = [
            # "Cevap: A" or "Yanıt: B" patterns
            rf"(?:cevap|yanıt|answer)\s*[:=]\s*([{labels}])",
            # "(A)" or "[A]" patterns
            rf"[\(\[]\s*([{labels}])\s*[\)\]]",
            # Standalone letter at start
            rf"^([{labels}])[\.\)\s,]",
            # Number patterns: "1", "2" etc.
            rf"(?:cevap|yanıt|answer)\s*[:=]\s*(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                val = match.group(1)
                if val.isdigit():
                    idx = int(val) - 1
                    if 0 <= idx < num_choices:
                        return idx
                else:
                    idx = labels.index(val.upper())
                    if 0 <= idx < num_choices:
                        return idx

        # Last resort: check if response is just a single letter
        if len(response) == 1 and response.upper() in labels:
            return labels.index(response.upper())

        return None

    @staticmethod
    def format_choices(choices: List[str]) -> str:
        """Format multiple choice options as labeled list."""
        labels = "ABCDEFGHIJ"
        lines = []
        for i, choice in enumerate(choices):
            if i < len(labels):
                lines.append(f"{labels[i]}. {choice}")
        return "\n".join(lines)
