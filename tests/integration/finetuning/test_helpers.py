"""Tests for maybe_empty_cache — migrated from legacy finetuning/test_helpers.py."""

import pytest


def maybe_empty_cache():
    """Clear CUDA cache if available — migrated from src.utils."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TestMaybeEmptyCache:
    def test_no_exception(self):
        try:
            maybe_empty_cache()
        except Exception as e:
            pytest.fail(f"maybe_empty_cache raised {e}")
