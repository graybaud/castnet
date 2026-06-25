"""Tests for maybe_empty_cache and other helpers."""
import pytest
from src.finetuning import maybe_empty_cache


class TestMaybeEmptyCache:
    def test_no_exception(self):
        """Should not raise regardless of CUDA availability."""
        try:
            maybe_empty_cache()
        except Exception as e:
            pytest.fail(f"maybe_empty_cache raised {e}")
