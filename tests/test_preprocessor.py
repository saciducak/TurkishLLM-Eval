"""Tests for preprocessor."""

import pytest
from turkishllm_eval.data.preprocessor import TurkishTextPreprocessor


class TestPreprocessor:
    def test_turkish_lower(self):
        assert TurkishTextPreprocessor.turkish_lower("I") == "ı"
        assert TurkishTextPreprocessor.turkish_lower("İ") == "i"
        assert TurkishTextPreprocessor.turkish_lower("İSTANBUL") == "istanbul"

    def test_turkish_upper(self):
        assert TurkishTextPreprocessor.turkish_upper("i") == "İ"
        assert TurkishTextPreprocessor.turkish_upper("ı") == "I"

    def test_extract_choice(self):
        assert TurkishTextPreprocessor.extract_choice("A") == 0
        assert TurkishTextPreprocessor.extract_choice("Cevap: B") == 1
        assert TurkishTextPreprocessor.extract_choice("(C)") == 2

    def test_clean_response(self):
        dirty = "<|endoftext|>Temiz yanıt</s>"
        clean = TurkishTextPreprocessor.clean_model_response(dirty)
        assert "endoftext" not in clean
        assert "Temiz yanıt" in clean

    def test_normalize(self):
        text = "  çok   fazla    boşluk  "
        result = TurkishTextPreprocessor.normalize(text)
        assert result == "çok fazla boşluk"
