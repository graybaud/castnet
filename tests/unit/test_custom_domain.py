"""Unit tests for custom domain evaluation."""

import pytest
from unittest.mock import MagicMock
from domain.metrics.custom_domain import (
    evaluate_custom_domain,
    CUSTOM_DOMAIN_QUESTIONS,
)


class TestEvaluateCustomDomain:
    def test_all_correct(self):
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = MagicMock(shape=(1, 10))
        tokenizer.decode.return_value = "1.2V"

        questions = [("What is the voltage?", "1.2V")]
        result = evaluate_custom_domain(model, tokenizer, questions)
        assert result["accuracy"] == 1.0
        assert result["correct"] == 1

    def test_all_wrong(self):
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = MagicMock(shape=(1, 10))
        tokenizer.decode.return_value = "xyz abc def"

        questions = [("What is the voltage?", "1.2V")]
        result = evaluate_custom_domain(model, tokenizer, questions)
        assert result["accuracy"] == 0.0

    def test_default_questions(self):
        assert len(CUSTOM_DOMAIN_QUESTIONS) == 15

    def test_empty_questions(self):
        model = MagicMock()
        tokenizer = MagicMock()
        result = evaluate_custom_domain(model, tokenizer, questions=[])
        assert result["accuracy"] == 0.0
        assert result["n_questions"] == 0
