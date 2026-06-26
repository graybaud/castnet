"""Tests for score I/O utilities."""

import os
import tempfile
import torch
import pytest
from infrastructure.persistence.score_io import (
    infer_method_from_filename,
    load_scores,
)


class TestInferMethod:

    def test_gradient(self):
        assert infer_method_from_filename("phi2_gradient_scores_b100.pt") == "gradient"

    def test_wanda(self):
        assert infer_method_from_filename("opt_wanda_scores.pt") == "wanda"

    def test_wanda_chain(self):
        assert infer_method_from_filename("model_wanda_chain_scores.pt") == "wanda_chain"

    def test_chain(self):
        assert infer_method_from_filename("model_chain_scores.pt") == "chain"

    def test_gps(self):
        assert infer_method_from_filename("model_gps_scores.pt") == "gps"

    def test_unknown(self):
        assert infer_method_from_filename("random.pt") == "unknown"

    def test_case_insensitive(self):
        assert infer_method_from_filename("MODEL_WANDA_SCORES.pt") == "wanda"


class TestLoadScores:

    @pytest.fixture
    def pt_scores(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "scores.pt")
        scores = {"fc1": torch.randn(10, 20), "fc2": torch.randn(20, 10)}
        torch.save(scores, path)
        yield path, scores
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def safetensors_scores(self):
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "scores.safetensors")
        scores = {"fc1": torch.randn(10, 20), "fc2": torch.randn(20, 10)}
        from safetensors.torch import save_file
        save_file(scores, path)
        yield path, scores
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_pt(self, pt_scores):
        path, expected = pt_scores
        loaded = load_scores(path)
        for name in expected:
            assert torch.equal(loaded[name], expected[name])

    def test_load_safetensors(self, safetensors_scores):
        path, expected = safetensors_scores
        loaded = load_scores(path)
        for name in expected:
            assert torch.equal(loaded[name], expected[name])

    def test_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_scores("/nonexistent/scores.pt")
