import pytest
"""
test_helpers.py — Tests for maybe_empty_cache.
"""

import torch
from src.evaluation import maybe_empty_cache


class TestMaybeEmptyCache:

    def test_no_exception(self):
        """Should not raise regardless of CUDA availability."""
        try:
            maybe_empty_cache()
        except Exception as e:
            pytest.fail(f"maybe_empty_cache raised {e}")
