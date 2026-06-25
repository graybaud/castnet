"""
test_score_io.py — Tests for load_scores.
"""

import os
import tempfile
import torch
import pytest
from src.extraction.utils.io import load_scores


class TestScoreIO:

    @pytest.fixture
    def tmp_scores(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "scores.pt")
        scores = {"fc1": torch.randn(10, 20), "fc2": torch.randn(20, 10)}
        torch.save(scores, path)
        yield path, scores
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_roundtrip(self, tmp_scores):
        path, expected = tmp_scores
        loaded = load_scores(path, device="cpu")
        assert len(loaded) == len(expected)
        for name in expected:
            assert torch.equal(loaded[name], expected[name])

    def test_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_scores("/nonexistent/scores.pt", device="cpu")

    def test_device_placement(self, tmp_scores):
        path, _ = tmp_scores
        loaded = load_scores(path, device="cpu")
        for t in loaded.values():
            assert t.device.type == "cpu"
