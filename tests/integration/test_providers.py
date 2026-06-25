"""Tests for data providers."""

import torch
from infrastructure.data.providers import C4Provider


class TestC4Provider:
    def test_imports(self):
        """C4Provider should be importable without crashing."""
        assert C4Provider is not None

    def test_has_get_batches(self):
        """C4Provider must implement BatchProvider interface."""
        assert hasattr(C4Provider, 'get_batches')
