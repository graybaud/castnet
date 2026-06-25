"""Tests for castnet.extraction.utils.printf"""
import sys, io
from src.utils import printf


class TestPrintf:
    def test_flush_default(self):
        """printf should flush by default."""
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        printf("test")
        sys.stdout = old_stdout
        assert "test" in buf.getvalue()

    def test_custom_kwargs(self):
        """printf should accept custom kwargs."""
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        printf("a", "b", sep="-")
        sys.stdout = old_stdout
        assert "a-b" in buf.getvalue()
