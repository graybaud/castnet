"""Tests for application coverage — updated for new use case names."""

import torch
from application.evaluate import EvaluateModelUseCase, EvalResult


class TestEvaluateModelUseCase:
    def test_eval_result_dataclass(self):
        result = EvalResult(
            perplexity=10.0,
            sparsity={"overall_sparsity_pct": 50.0},
            total_tokens=1000,
        )
        assert result.perplexity == 10.0
