"""
test_diagnose.py — Smoke tests for diagnose_scores.
"""

import torch
from src.extraction.eval.diagnose import diagnose_scores


class TestDiagnoseScores:

    def test_mixed(self):
        scores = {"fc1": torch.rand(5, 10), "fc2": torch.zeros(10, 5)}
        diagnose_scores(scores, max_layers=5)  # Should not raise

    def test_all_zero(self):
        diagnose_scores({"fc": torch.zeros(3, 4)})  # Should not raise

    def test_empty(self):
        diagnose_scores({})  # Should not raise
