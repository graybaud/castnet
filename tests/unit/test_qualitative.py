"""
test_qualitative.py — Smoke tests for qualitative generation.
Migrated from legacy.
"""

import pytest
import torch
from domain.metrics.qualitative import (
    run_greedy_tests,
    run_sampled_tests,
    analyze_repetition,
    QUALITATIVE_PROMPTS,
)


class TestQualitativePrompts:
    def test_prompts_not_empty(self):
        assert len(QUALITATIVE_PROMPTS) >= 3


class TestAnalyzeRepetition:
    def test_unique_text(self):
        result = analyze_repetition("hello world this is a test")
        assert result["unique_ratio"] == 1.0

    def test_repeated_text(self):
        result = analyze_repetition("hello hello hello world")
        assert result["unique_ratio"] < 1.0

    def test_empty_text(self):
        result = analyze_repetition("")
        assert result["unique_ratio"] == 1.0
